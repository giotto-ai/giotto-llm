import json
import os
from functools import partial
from typing import Any, Literal

import torch
from accelerate import PartialState
from qwen_vl_utils import process_vision_info
from torch import Tensor, nn
from torch.utils.data import default_collate
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
)
from transformers.utils import ModelOutput

from ..prompts.grid_formatter import GridFormatter
from ..transforms import Transforms, _BackTransformTestOutput
from ..type_aliases import JSONTask, OAIMessage
from ..utils import BNBConfig, is_launched_with_torchrun
from ..wrapper import ModelWrapper


class MolmoWrapper(ModelWrapper):
    """Wrapper for Qwen2VL models."""

    model_type = "image-text-to-text"

    def __init__(
        self,
        model_id: str = "allenai/MolmoE-1B-0924",
        gpu_index: int = 0,
        quantization: (
            Literal[
                "no",
                "4bit-nf4",
                "4bit-dq-nf4",
                "4bit-fp4",
                "4bit-dq-fp4",
                "8bit-6",
                "8bit-5",
                "8bit-4",
            ]
            | None
        ) = None,
        config: dict[str, Any] = {},
    ):
        if quantization is None:
            quantization = "4bit-fp4"
        # Increase the max supported number of tokens
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_config.max_position_embeddings = max(
            model_config.max_position_embeddings, self._min_max_position_embeddings
        )
        # Merge default config with the given one
        config = {
            "quantization_config": BNBConfig[quantization],
            "device_map": (
                {"": PartialState().process_index}
                if is_launched_with_torchrun()
                else {"": f"cuda:{gpu_index}"}
            ),
            "config": model_config,
            "torch_dtype": "auto",  # If set to None, this can cause nan issues in log-likelihoods (due to fp16)
            "trust_remote_code": True,
        } | config

        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **config,
        )
        # This is a bit hacky since the processor isn't loaded correctly from pre-trained models
        self.processor = AutoProcessor.from_pretrained(
            model_config.auto_map["AutoConfig"].split("--")[0],
            trust_remote_code=True,
            pad_token="|<PADDING>|",
        )
        self.tokenizer = self.processor.tokenizer
        self._replace_custom_embedding()
        self.generation_prompt_token_ids = self._get_generation_prompt_token_ids()
        self._init_grid_formatter_and_update_tokenizer_model()
        self._set_output_token_ids()
        self._init_sanity_checks()

    def _replace_custom_embedding(self) -> None:
        """Convert Molmo custom embedding to standard nn.Embedding.

        Molmo uses a custom embedding that breaks a lot of standard
        huggingface calls.
        """
        config = self.model.config
        input_embeddings = self.model.get_input_embeddings()
        assert self.model.model.transformer.wte is input_embeddings
        assert not isinstance(input_embeddings, nn.Embedding)
        weights = torch.cat([input_embeddings.embedding, input_embeddings.new_embedding], dim=0)
        self.model.model.transformer.wte = nn.Embedding(
            num_embeddings=weights.shape[0],
            embedding_dim=weights.shape[1],
            padding_idx=self.tokenizer.pad_token_id,
        )
        self.model.model.transformer.wte.weight.data = weights.data

    def _get_generation_prompt_token_ids(self) -> Tensor:
        """Determine what the generation prompt token ids are"""
        generation_prompt_ids: Tensor = self.processor.process(text="")["input_ids"][-2:]
        return generation_prompt_ids

    def _generate(
        self,
        batch_inputs: dict[str, Tensor],
        generation_config: GenerationConfig,
        constraints: list[dict[str, Any],],
        constraints_strategy: Literal["no", "token_subset", "valid"] = "valid",
        constraints_use_verifier_shapes: bool = True,
        constraints_use_verifier_colors: bool = True,
        constraints_use_verifier_pixels: bool = True,
    ) -> ModelOutput:
        output: ModelOutput = self.model.generate_from_batch(
            batch_inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
            prefix_allowed_tokens_fn=partial(
                self.prefix_allowed_tokens_fn,
                input_size=batch_inputs["input_ids"].shape[1],
                # batch_constraints=constraints,
            ),
        )
        return output


    def _conversations_to_batch_inputs(self, conversations: list[OAIMessage]) -> dict[str, Tensor]:
        if len(conversations) > 1:
            raise RuntimeError("Molmo models only works with batch size of 1.")
        # Use the Qwen2 preprocessing, which will apply the image_scale_factor
        image_inputs = process_vision_info(conversations)[0]
        # Create text to encode in the internal molmo format.
        # Molmo is not trained using system messages, so concatenate system and user messages.
        if len(conversations[0]) == 2:
            text_inputs = [
                " User: "
                + (system["content"][0]["text"] + " " + user["content"][0]["text"]).lstrip()
                + " Assistant:"
                for system, user in conversations
            ]
        else:
            assert len(conversations[0]) == 3
            text_inputs = [
                " User: "
                + (system["content"][0]["text"] + " " + user["content"][0]["text"]).lstrip()
                + " Assistant: "
                + assistant["content"][0]["text"]
                + self.tokenizer.eos_token
                for system, user, assistant in conversations
            ]

        text_tokens = [
            self.tokenizer.encode(text, add_special_tokens=False) for text in text_inputs
        ]
        inputs = [
            self.processor.process(images=[image], tokens=tokens)
            for image, tokens in zip(image_inputs, text_tokens, strict=True)
        ]

        batch_inputs: dict[str, Tensor] = default_collate(inputs)
        return batch_inputs

    def collate_fn_eval(
        self,
        examples: list[tuple[OAIMessage, JSONTask, dict[str, Any], int, _BackTransformTestOutput]],
    ) -> dict[str, dict[str, Any] | list[int] | list[_BackTransformTestOutput] | list[JSONTask]]:
        """Data collator to encode text and image pairs."""
        conversations = [example[0][:2] for example in examples]
        batch_inputs = self._conversations_to_batch_inputs(conversations)

        batch_indices = [example[2] for example in examples]
        backtransforms = [example[3] for example in examples]
        transformed_tasks = [example[1] for example in examples]

        return {
            "batch_inputs": batch_inputs,
            "batch_indices": batch_indices,
            "backtransforms": backtransforms,
            "transformed_tasks": transformed_tasks,
        }

    def collate_fn_train(
        self,
        examples: list[tuple[OAIMessage, JSONTask, int, _BackTransformTestOutput]],  # type: ignore
        mask_inputs: bool = True,
    ) -> dict[str, Tensor]:
        """The collate function."""
        assert all(example[0][2]["content"][0]["text"] != "" for example in examples)

        conversations = [example[0] for example in examples]
        batch = self._conversations_to_batch_inputs(conversations)
        batch["labels"] = self._create_labels(batch["input_ids"])
        return batch
