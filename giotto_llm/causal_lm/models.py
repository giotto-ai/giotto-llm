import json
import os
from typing import Any, Literal

import torch
from accelerate import PartialState
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..prompts.grid_formatter import GridFormatter
from ..transforms import _BackTransformTestOutput
from ..type_aliases import JSONTask, OAIMessage
from ..utils import BNBConfig, is_launched_with_torchrun
from ..wrapper import ModelWrapper
from transformers import PreTrainedTokenizerFast


class CausalLMWrapper(ModelWrapper):
    """Wrapper for CausalLM models."""

    model_type = "text-to-text"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B",
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
        online_finetuning: bool = False,
    ):
        if quantization is None:
            quantization = "8bit-6"
        # Increase the max supported number of tokens
        model_config = AutoConfig.from_pretrained(model_id)
        model_config.max_position_embeddings = max(
            model_config.max_position_embeddings, self._min_max_position_embeddings
        )
        # Merge default config with the given one
        config = {
            "quantization_config": BNBConfig[quantization],
            "device_map": (
                {"": PartialState().local_process_index}
                if is_launched_with_torchrun()
                else (
                    {"": f"cuda:{gpu_index}"}
                    if not online_finetuning
                    else {"": torch.cuda.current_device()}  # type: ignore
                )
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
        # Load your custom minimal tokenizer instead of AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(tokenizer.chat_template)


        # Get the directory of the current script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the minimal_tokenizer.json file
        tokenizer_path = os.path.join(current_directory, "minimal_tokenizer.json")

        # Initialize the tokenizer with the constructed path
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        # Define pad and eos tokens if they aren't auto-detected:
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.eos_token = "<eos>"

        # After setting the tokenizer, resize model embeddings to match the smaller vocab:
        self.model.resize_token_embeddings(len(self.tokenizer))
        chat_template = {
        "system": "You are a helpful assistant.",
        "user": "{user_input}",
        "assistant": "{assistant_response}"
    }
        self.tokenizer.chat_template = chat_template


        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            # Handle special cases here instead of subclassing for 1 line
            try:
                self.tokenizer.pad_token = next(
                    token
                    for token in self.tokenizer.get_added_vocab()
                    if "finetune" in token or "<|video_pad|>" in token
                )
                self.tokenizer.pad_token_id = self.tokenizer.vocab[self.tokenizer.pad_token]
            except StopIteration:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.generation_prompt_token_ids = torch.tensor([128006, 78191, 128007, 271])
        self._init_grid_formatter_and_update_tokenizer_model()
        self._init_data_config()
        self._set_output_token_ids()
        self._init_sanity_checks()

    def collate_fn_eval(
        self,
        examples: list[tuple[OAIMessage, JSONTask, dict[str, Any], int, _BackTransformTestOutput]],
    ) -> dict[str, dict[str, Any] | list[int] | list[_BackTransformTestOutput] | list[JSONTask]]:
        """The collate function."""
        conversation = [example[0][:2] for example in examples]
        encoded_conversation = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch_inputs = self.tokenizer(
            encoded_conversation,
            padding=True,
            truncation=False,
            split_special_tokens=False,
            return_tensors="pt",
        )
        transformed_tasks = [example[1] for example in examples]
        constraints = [example[2] for example in examples]
        batch_indices = [example[3] for example in examples]
        backtransforms = [example[4] for example in examples]
        return {
            "batch_inputs": batch_inputs,
            "batch_indices": batch_indices,
            "backtransforms": backtransforms,
            "transformed_tasks": transformed_tasks,
            "constraints": constraints,
        }

    def collate_fn_train(
        self,
        examples: list[tuple[OAIMessage, JSONTask, dict[str, Any], int, _BackTransformTestOutput]],
        mask_inputs: bool = True,
    ) -> dict[str, Tensor]:
        """The collate function."""
        assert all(example[0][2]["content"] != "" for example in examples)

        conversation = [example[0] for example in examples]
        encoded_conversation = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        if self.tokenizer.eos_token not in encoded_conversation[0]:
            encoded_conversation = [
                conv + self.tokenizer.eos_token for conv in encoded_conversation
            ]
        batch: dict[str, Tensor] = self.tokenizer(
            encoded_conversation,
            padding=True,
            truncation=False,
            split_special_tokens=False,
            return_tensors="pt",
        )
        batch["labels"] = self._create_labels(batch["input_ids"], mask_inputs=mask_inputs)
        return batch
