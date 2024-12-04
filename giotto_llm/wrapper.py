import copy
import gc
import json
import logging
import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Any, Literal, Optional, Type

import numpy as np
import psutil
import torch
import transformers
from pydantic import BaseModel, Field, ValidationError, validator
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.utils import ModelOutput

from .consts import DATA_CONFIG_FILENAME, GRID_FORMATTER_CONFIG_FILENAME, MAX_GRID_SIZE
from .data import Dataset
from .prompts.consts import TYPES_OF_PROMPTS
from .prompts.grid_formatter import GridFormatter
from .selection import select_top_n  # type: ignore
from .transforms import Transforms, _BackTransformTestOutput, backtransform_test_output
from .type_aliases import Attempts, Grid, JSONTask, OAIMessage
from .utils import RepeatSampler, write_json


class EvaluationConfig(BaseModel):
    """Config for customizing evaluation behavior.

    Args:
        n_attempts: how many attempts to return. If `None` returns all.
        n_transforms: do multiple predictions per task with a different transform.
            Requires that random transforms is used.
        batch_size: the batch size. Batch sizes different than 1 can change the result
            due to numerical stability, so test for each model.
        n_dataloader_workers: number of worker subprocesses used by the DataLoader
        constraints_strategy: the strategy used to constrain the allowed tokens during generation.
            'no' will use no constraints. 'token_subset' will only use the 15 possible output tokens.
            'valid' will enforce valid grids.
        constraints_use_verifier_shapes: use the shapes constraint from verifier
        constraints_use_verifier_colors: use the colors constraint from verifier
        constraints_use_verifier_pixels: use the pixels constraint from verifier
        input_tokens_limit: limit the number of tokens of the training examples and test input, by subsampling
            the training examples.
        save_generation_metadata: if `True` save additional metadata in './generation_metadata'.

        selection_weights_method: the method used to weigh each attempt during selection
        selection_threshold: the score threshold in [0, 1] to switch between voting strategies
        generation_config: parameters passed to the model.generate method. Typically used to set
            generation/sampling strategy via `num_beams`, `temperature` etc.. `num_return_sequences=n` will
            generate `n` responses per input.
    """

    n_attempts: int | None = Field(ge=1, default=None)
    n_transforms: int = Field(ge=1, default=1)
    batch_size: int = Field(ge=1, default=1)
    n_dataloader_workers: int = Field(ge=1, default=psutil.cpu_count(logical=False))  # type: ignore
    save_generation_metadata: bool = False
    constraints_strategy: Literal["no", "token_subset", "valid"] = "valid"
    constraints_use_verifier_shapes: bool = True
    constraints_use_verifier_colors: bool = True
    constraints_use_verifier_pixels: bool = True
    input_tokens_limit: int | None = None
    selection_weights_method: Literal["uniform", "ll_sum", "entropy"] = "ll_sum"
    selection_threshold: float = 0.5
    generation_config: dict[str, Any] = {"max_new_tokens": 1024, "num_return_sequences": 1}
    use_majority_vote: bool = False

    class Config:
        protected_namespaces = ()
        extra = "forbid"

    @validator("generation_config")
    def check_generation_config(
        cls: "EvaluationConfig", value: dict[str, Any], values: dict[str, Any]
    ) -> dict[str, Any]:
        # These keys need to have specific values
        for key in ("return_dict_in_generate", "output_scores"):
            if key in value:
                raise ValueError(f"Cannot set {key} in generation_config")
        return value

    @validator("constraints_use_verifier_shapes")
    def check_constraints_use_verifier_shapes(
        cls: "EvaluationConfig", value: bool, values: dict[str, Any]
    ) -> bool:
        if values["constraints_strategy"] != "valid" and value is True:
            raise ValueError("use_verifier_shapes=True requires constraints_strategy='valid'")
        return value

    @validator("constraints_use_verifier_colors")
    def check_constraints_use_verifier_colors(
        cls: "EvaluationConfig", value: bool, values: dict[str, Any]
    ) -> bool:
        if values["constraints_strategy"] != "valid" and value is True:
            raise ValueError("use_verifier_colors=True requires constraints_strategy='valid'")
        return value

    @validator("constraints_use_verifier_pixels")
    def check_constraints_use_verifier_pixels(
        cls: "EvaluationConfig", value: bool, values: dict[str, Any]
    ) -> bool:
        if values["constraints_strategy"] != "valid" and value is True:
            raise ValueError("use_verifier_pixels=True requires constraints_strategy='valid'")
        return value


class ModelWrapper:
    """Baseclass for a unified interface between models.

    Assumes an underlying PyTorch model.
    """

    # Attributes required in children
    _target_modules: list[str] | str = "all-linear"  # for Lora configuation
    _min_max_position_embeddings = 16384  # Enough to cover expected number of tokens
    model_id: str
    model_type: Literal["image-text-to-text", "text-to-text"]  # for data format
    model: transformers.modeling_utils.PreTrainedModel
    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    grid_formatter: GridFormatter
    generation_prompt_token_ids: Tensor  # for masking labels during training
    output_token_ids: dict[str | int, int]
    _data_config: dict[str, bool | str]

    def __init__(
        self,
        model_id: str = "",
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
        raise NotImplementedError

    @property
    def data_config(self) -> dict[str, bool | str]:
        return self._data_config

    @data_config.setter
    def data_config(self, value: dict[str, bool | str]) -> None:
        assert len(value) == 3
        for key in ("compress_colors", "transform_background_color", "prompt_type"):
            assert key in value
        self._data_config = value

    def save_pretrained(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        self.grid_formatter.save(output_path)
        if hasattr(self, "processor"):
            self.processor.save_pretrained(output_path)
        write_json(self.data_config, f"{output_path}/{DATA_CONFIG_FILENAME}")

    def evaluate(
        self,
        tasks: dict[str, JSONTask],
        logger: logging.Logger,
        config: EvaluationConfig = EvaluationConfig(),
    ) -> dict[str, Attempts]:
        """Make predictions for all tasks."""
        self.model.eval()
        limit_colors = self.data_config["compress_colors"]
        assert isinstance(limit_colors, bool)
        if config.n_transforms == 1:
            transforms = Transforms(
                test=False,
                order=None,
                color=None,
                limit_colors=limit_colors,
                rigid=False,
                max_tokens=config.input_tokens_limit,
            )
        else:
            transforms = Transforms(
                test=False,
                order="reorder",
                color=(
                    "all"
                    if self.data_config["transform_background_color"] is True
                    else "foreground"
                ),
                limit_colors=limit_colors,
                rigid=True,
                max_tokens=config.input_tokens_limit,
            )
        prompt_type = self.data_config["prompt_type"]
        assert isinstance(prompt_type, str)
        messages_fn = TYPES_OF_PROMPTS[prompt_type](grid_formatter=self.grid_formatter)
        task_grids: dict[str, list[Grid]] = defaultdict(list)
        task_log_probs: dict[str, list[float]] = defaultdict(list)
        original_tasks: dict[str, JSONTask] = {}
        task_constraints: dict[str, dict[str, Any]] = {}
        dataset = Dataset(
            tasks=tasks,
            messages_fn=messages_fn,
            model_type=self.model_type,
            transforms=transforms,
            constraints_strategy=config.constraints_strategy,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=RepeatSampler(config.n_transforms, len(dataset)),
            num_workers=config.n_dataloader_workers,
            collate_fn=self.collate_fn_eval,
            pin_memory=True,
            drop_last=False,
        )
        generation_config = GenerationConfig(
            return_dict_in_generate=True,
            output_scores=True,
            **config.generation_config,
        )
        for i, batch in enumerate(dataloader):
            logger.info(f"Generating batch {i+1} of {len(dataloader)}")
            batch_indices = batch["batch_indices"]
            constraints = batch["constraints"]
            task_ids = [dataset.keys[task_index] for task_index in batch_indices]
            for idx, task_id in enumerate(task_ids):
                original_tasks[task_id] = dataset.tasks[task_id]
                task_constraints[task_id] = constraints[idx]
            logger.info(f"Analyzing tasks {task_ids=}")
            batch_inputs = batch["batch_inputs"]
            for key in batch_inputs:
                batch_inputs[key] = batch_inputs[key].to(device=self.model.device)
            input_ids = batch_inputs["input_ids"].to(device=self.model.device)

            backtransforms = batch["backtransforms"]
            if generation_config.num_return_sequences > 1:
                batch_indices = [
                    i for i in batch_indices for _ in range(generation_config.num_return_sequences)
                ]
                backtransforms = [
                    t for t in backtransforms for _ in range(generation_config.num_return_sequences)
                ]

            output = self._generate(
                batch_inputs,
                generation_config,
                constraints=constraints,
                constraints_strategy=config.constraints_strategy,
                constraints_use_verifier_shapes=config.constraints_use_verifier_shapes,
                constraints_use_verifier_colors=config.constraints_use_verifier_colors,
                constraints_use_verifier_pixels=config.constraints_use_verifier_pixels,
            )
            responses = self._decode(output.sequences, input_size=input_ids.shape[1])
            log_probs = self._get_log_likelihoods(output, input_size=input_ids.shape[1])
            del batch_inputs
            del input_ids
            del output
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            attempts = [
                self.grid_formatter.decode_grid(
                    str_containing_grid=response,
                    input_or_output="output",
                    logger=logger,
                )
                for response in responses
            ]

            for attempt, log_prob, idx, backtransform in zip(
                attempts, log_probs, batch_indices, backtransforms, strict=True
            ):
                if attempt is None:
                    continue
                task_id = dataset.keys[idx]
                task_grids[task_id].append(
                    backtransform_test_output(grid=attempt, backtransform=backtransform)
                )
                task_log_probs[task_id].append(log_prob)

        results: dict[str, Attempts] = self._create_results(
            task_attempts=task_grids,
            n_attempts=config.n_attempts,
            task_log_probs=task_log_probs,
            tasks=original_tasks,
            weight_method=config.selection_weights_method,
            threshold=config.selection_threshold,
            constraints=task_constraints,
            constraints_strategy=config.constraints_strategy,
        )

        return results

    def collate_fn_eval(
        self,
        examples: list[tuple[OAIMessage, JSONTask, dict[str, Any], int, _BackTransformTestOutput]],
    ) -> dict[str, dict[str, Any] | list[int] | list[_BackTransformTestOutput] | list[JSONTask]]:
        raise NotImplementedError

    def collate_fn_train(
        self,
        examples: list[tuple[OAIMessage, JSONTask, dict[str, Any], int, _BackTransformTestOutput]],
        mask_inputs: bool = True,
    ) -> dict[str, Tensor]:
        raise NotImplementedError

    def untie_word_embeddings(self) -> None:
        if self.model.config.tie_word_embeddings is True:
            input_embeddings = self.model.get_input_embeddings()
            self.model.lm_head.weight = nn.Parameter(input_embeddings.weight.data.clone())
            self.model.config.tie_word_embeddings = False
            assert self.model.lm_head.weight is not input_embeddings.weight
            if isinstance(self._target_modules, list):
                assert hasattr(self.model.model, "embed_tokens")
                self._target_modules = list(
                    set(self._target_modules) | set(("lm_head", "embed_tokens"))
                )

    def _save_generation_metadata(
        self,
        task_ids: list[str],
        output: dict[str, Any],
        transformed_tasks: list[JSONTask],
        responses: list[str],
        attempts: list[Grid | None],
        backtransforms: list[_BackTransformTestOutput],
        input_size: int,
    ) -> None:
        os.makedirs("./generation_metadata", exist_ok=True)
        num_generated_attempts = len(task_ids)
        batch_size = len(transformed_tasks)
        num_return_sequences = num_generated_attempts // batch_size
        token_ids = torch.tensor(list(self.output_token_ids.values()))
        for i in range(num_generated_attempts):
            task_id = task_ids[i].replace("-|-", "_test_idx_")
            for j in range(100):
                filename = f"./generation_metadata/{task_id}_transform_{j}.pickle"
                if os.path.exists(filename):
                    continue
                break
            seq_length = len(output["scores"])
            data = {
                "transformed_task": transformed_tasks[i // num_return_sequences],
                "response": responses[i],
                "attempt": attempts[i],
                "sequences": output["sequences"][i][input_size:],
                "logits": torch.stack(
                    [output["scores"][idx][i][token_ids] for idx in range(seq_length)]
                ),
                "backtransform": backtransforms[i],
            }
            if "hidden_states" in output:
                data["last_hidden_state"] = (
                    torch.cat(
                        [output["hidden_states"][idx][-1][i] for idx in range(seq_length)], dim=0
                    )[input_size:],
                )
            with open(filename, "wb") as f:
                pickle.dump(data, f)

    def _init_data_config(self) -> None:
        # We read the data config if possible.
        # Otherwise, we use the default init values
        path_to_data_config = os.path.join(self.model_id, DATA_CONFIG_FILENAME)
        if os.path.exists(path_to_data_config):
            with open(path_to_data_config, "r") as f:
                data_config = json.load(f)
        else:
            data_config = {
                "compress_colors": False,
                "transform_background_color": True,
                "prompt_type": "prompt_solve_short",
            }
        self.data_config = data_config

    def _init_sanity_checks(self) -> None:
        for attr in (
            "_target_modules",
            "_min_max_position_embeddings",
            "model_type",
            "model",
            "tokenizer",
            "grid_formatter",
            "generation_prompt_token_ids",
            "output_token_ids",
            "_data_config",
        ):
            assert hasattr(self, attr)
        assert self.tokenizer.pad_token is not None
        assert self.tokenizer.pad_token != self.tokenizer.eos_token

    def _init_grid_formatter_and_update_tokenizer_model(self) -> None:
        self._init_grid_formatter()

        # Update tokenizer special tokens and model embedding according to special tokens
        additional_special_tokens = self.grid_formatter.get_special_tokens_not_in(
            tokenizer=self.tokenizer
        )
        new_special_tokens = {"additional_special_tokens": additional_special_tokens}
        self.tokenizer.add_special_tokens(new_special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _init_grid_formatter(self) -> None:
        # We build a GridFormatter from a config, if present
        # Otherwise, we use the default init values
        path_to_grid_formatter_config = os.path.join(self.model_id, GRID_FORMATTER_CONFIG_FILENAME)
        if os.path.exists(path_to_grid_formatter_config):
            with open(path_to_grid_formatter_config, "r") as f:
                grid_formatter_config = json.load(f)
            self.grid_formatter = GridFormatter(**grid_formatter_config)
        else:
            self.grid_formatter = GridFormatter()

    def _generate(
        self,
        batch_inputs: dict[str, Tensor],
        generation_config: GenerationConfig,
        constraints: list[dict[str, Any]],
        constraints_strategy: Literal["no", "token_subset", "valid"] = "no",
        constraints_use_verifier_shapes: bool = False,
        constraints_use_verifier_colors: bool = False,
        constraints_use_verifier_pixels: bool = False,
    ) -> ModelOutput:
        output: ModelOutput = self.model.generate(
            **batch_inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
        )
        return output

    def _set_output_token_ids(self) -> None:
        self.output_token_ids: dict[str | int, int] = {
            "start_output": self.tokenizer.vocab[self.grid_formatter.sO_token],
            "end_output": self.tokenizer.vocab[self.grid_formatter.eO_token],
            "start_row": self.tokenizer.vocab[self.grid_formatter.sR_token],
            "end_row": self.tokenizer.vocab[self.grid_formatter.eR_token],
            "eos": self.tokenizer.eos_token_id,
            # "eot": 128001,
            0: self.tokenizer.vocab[self.grid_formatter.c0],
            1: self.tokenizer.vocab[self.grid_formatter.c1],
            2: self.tokenizer.vocab[self.grid_formatter.c2],
            3: self.tokenizer.vocab[self.grid_formatter.c3],
            4: self.tokenizer.vocab[self.grid_formatter.c4],
            5: self.tokenizer.vocab[self.grid_formatter.c5],
            6: self.tokenizer.vocab[self.grid_formatter.c6],
            7: self.tokenizer.vocab[self.grid_formatter.c7],
            8: self.tokenizer.vocab[self.grid_formatter.c8],
            9: self.tokenizer.vocab[self.grid_formatter.c9],
        }

    def prefix_allowed_tokens_fn(
        self,
        batch_id: int,
        input_ids: Tensor,
        input_size: int,
        batch_constraints: list[dict[str, Any]],
        strategy: str,
        use_verifier_shapes: bool,
        use_verifier_colors: bool,
        use_verifier_pixels: bool,
    ) -> list[int]:
        """Restrict the allowed tokens to generate"""
        if strategy == "no":
            return list(self.tokenizer.vocab.values())
        if strategy == "token_subset":
            return list(self.output_token_ids.values())
        assert strategy == "valid"

        constraints = batch_constraints[batch_id]
        output_ids = input_ids[input_size:]
        output_size = output_ids.shape[0]
        # Start with start_output, start_row
        if output_size == 0:
            return [self.output_token_ids["start_output"]]
        if output_size == 1:
            return [self.output_token_ids["start_row"]]

        # End with end_output, eos
        if self.output_token_ids["end_output"] in output_ids:
            return [
                self.output_token_ids["eos"]
            ]  # [self.output_token_ids["eos"], self.output_token_ids["eot"]]
        else:
            allowed_tokens = copy.copy(self.output_token_ids)
            allowed_tokens.pop("eos", None)
            # allowed_tokens.pop("eot", None)

        # Force valid grid sizes
        # Make sure there's no empty rows returned
        if output_ids[-1] == self.output_token_ids["start_row"]:
            allowed_tokens.pop("end_output", None)
            allowed_tokens.pop("end_row", None)
            allowed_tokens.pop("start_row", None)

        ## end_row restricts possible next tokens
        if output_ids[-1] == self.output_token_ids["end_row"]:
            for key in [k for k in allowed_tokens if k not in ("start_row", "end_output")]:
                allowed_tokens.pop(key, None)

        start_row_indices = torch.where(output_ids == self.output_token_ids["start_row"])[
            0
        ].tolist()
        n_rows = len(start_row_indices)
        # end_row_indices = torch.where(output_ids == self.output_token_ids["end_row"])[0]
        # if n_rows > len(end_row_indices):
        #    allowed_tokens.pop("start_row", None)

        # max MAX_GRID_SIZE rows
        if n_rows == MAX_GRID_SIZE:
            allowed_tokens.pop("start_row", None)

        if n_rows == 1:
            first_row_size = 0
            current_row_start_idx = start_row_indices[0]
        else:
            first_row_start_idx = start_row_indices[0]
            first_row_end_idx = torch.where(output_ids == self.output_token_ids["end_row"])[0][
                0
            ].item()
            first_row_size = first_row_end_idx - first_row_start_idx - 1
            current_row_start_idx = start_row_indices[-1]
        current_row_size: int = output_size - current_row_start_idx - 1

        # max MAX_GRID_SIZE columns
        if current_row_size == MAX_GRID_SIZE and output_ids[-1] != self.output_token_ids["end_row"]:
            return [self.output_token_ids["end_row"]]

        # once the number of columns has been generated, only allow
        # new rows to have the same number of columns
        if first_row_size > 0:
            if current_row_size < first_row_size:
                allowed_tokens.pop("end_row", None)
            elif output_ids[-1] == self.output_token_ids["end_row"]:
                # A bit hacky but makes the blow shape logic work
                current_row_size -= 1
            else:
                return [self.output_token_ids["end_row"]]

        # restrict shapes
        if use_verifier_shapes is True and len(constraints["size"]) > 0:
            possible_n_rows = [shape[0] for shape in constraints["size"]]
            possible_n_columns = [shape[1] for shape in constraints["size"]]
            if n_rows not in possible_n_rows:
                allowed_tokens.pop("end_output", None)
            if current_row_size not in possible_n_columns:
                allowed_tokens.pop("end_row", None)
            if n_rows == max(possible_n_rows):
                allowed_tokens.pop("start_row", None)
            if (
                current_row_size == max(possible_n_columns)
                and output_ids[-1] != self.output_token_ids["end_row"]
            ):
                return [self.output_token_ids["end_row"]]

        # restrict colors
        if use_verifier_colors is True and len(constraints["colors"]) > 0:
            for key in [k for k in range(10) if k not in constraints["colors"]]:
                allowed_tokens.pop(key, None)

        next_color_pixel_indices = (n_rows - 1, current_row_size - 0)
        if use_verifier_pixels is True and next_color_pixel_indices in constraints["pixels"]:
            # for typing
            color = constraints["pixels"][next_color_pixel_indices]
            return [self.output_token_ids[color]]

        return list(allowed_tokens.values())

    def _create_results(
        self,
        task_attempts: dict[str, list[Grid]],
        constraints: dict[str, dict[str, Any]],
        task_log_probs: dict[str, list[float]],
        tasks: dict[str, JSONTask],
        n_attempts: int | None,
        weight_method: Optional[Literal["uniform", "ll_sum", "entropy"]],
        threshold: float,
        constraints_strategy: Literal["no", "token_subset", "valid"],
    ) -> dict[str, Attempts]:
        """Sort attempts by log-likelihood, merge and combine test examples from same tasks"""
        results: dict[str, Attempts] = defaultdict(lambda: defaultdict(list))
        for split_task_id in sorted(task_attempts.keys()):
            if "-|-" in split_task_id:
                tokens = split_task_id.split("-|-")
                task_id = tokens[0]
                test_idx = int(tokens[1])
            else:
                task_id = split_task_id
                test_idx = 0

            if constraints_strategy == "valid":
                print("Using ranking and filtering approach..")
                results[task_id][test_idx] = select_top_n(
                    attempts=task_attempts[split_task_id],
                    log_probs=task_log_probs[split_task_id],
                    task=tasks[split_task_id],
                    weight_method=weight_method,
                    threshold=threshold,
                    constraints=constraints[split_task_id],
                    n_attempts=n_attempts,
                )

            else:
                attempts = task_attempts[split_task_id]
                # There can be duplicate attempts, so mean the log likelihood of duplicates
                attempt_log_likelihoods: dict[str, list[float]] = defaultdict(list)
                for i, attempt in enumerate(attempts):
                    attempt_log_likelihoods[str(attempt)].append(task_log_probs[split_task_id][i])

                grids = [json.loads(attempt) for attempt in attempt_log_likelihoods.keys()]
                log_likelihoods = [np.mean(ll) for ll in attempt_log_likelihoods.values()]

                idx = np.argsort(log_likelihoods)[::-1]
                if n_attempts is not None:
                    idx = idx[:n_attempts]
                results[task_id][test_idx] = [grids[i] for i in idx]
        return results

    def _decode(self, output_ids: Tensor, input_size: int) -> list[str]:
        response: list[str] = self.tokenizer.batch_decode(
            output_ids[:, input_size:],
            skip_special_tokens=False,
        )
        return response

    def _get_log_likelihoods(self, output: GenerateDecoderOnlyOutput, input_size: int) -> Tensor:
        # Remove input tokens, as well as start/end tokens.
        generated_tokens = output.sequences[:, input_size:].cpu().detach()
        # Stack logits to get shape [batch_size, sequence_length, vocab_size]
        logits = torch.stack(output.scores, dim=1).cpu().detach()
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        # Get attention mask (1s for real tokens, 0s for padding)
        attention_mask = (generated_tokens != self.tokenizer.pad_token_id).long()
        # Select log probabilities for the generated tokens
        log_likelihoods = log_probs.gather(2, generated_tokens.unsqueeze(-1)).squeeze(-1)
        # Apply attention mask to ignore padding in log likelihood
        masked_log_likelihoods = log_likelihoods * attention_mask
        # Compute total log likelihood (sum across all tokens in the sequence)
        total_log_likelihood: Tensor = masked_log_likelihoods.sum(dim=-1)
        return total_log_likelihood

    def _get_generation_prompt_token_ids(self) -> Tensor:
        """Determine what the generation prompt token ids are"""
        no_prompt_ids = torch.tensor(
            self.tokenizer.apply_chat_template(
                conversation=[[{"role": ""}]],
                tokenize=True,
                add_generation_prompt=False,
            )[0]
        )
        prompt_ids = torch.tensor(
            self.tokenizer.apply_chat_template(
                conversation=[[{"role": ""}]],
                tokenize=True,
                add_generation_prompt=True,
            )[0]
        )
        assert (no_prompt_ids == prompt_ids[: no_prompt_ids.shape[0]]).all()
        generation_prompt_ids = prompt_ids[no_prompt_ids.shape[0] :]
        return generation_prompt_ids

    def _get_input_mask(self, tokens: Tensor) -> Tensor:
        """Get a mask for input tokens

        Get a boolean mask that is True everywhere preceding (and including)
        the generation prompt tokens.
        """
        # This can probably be vectorized, but should be fast enough
        n_generation_prompt_tokens = self.generation_prompt_token_ids.shape[0]
        # Find tokens that match the first generation prompt token
        indices = torch.where(tokens == self.generation_prompt_token_ids[0])
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, j in zip(*indices, strict=True):
            # For each hit, check if the rest of the generation prompt token ids follows
            if (
                tokens[i, j : j + n_generation_prompt_tokens] == self.generation_prompt_token_ids
            ).all():
                # mask the generation prompt and everything preceding it
                mask[i, : j + n_generation_prompt_tokens] = True
        return mask

    def _create_labels(self, tokens: Tensor, mask_inputs: bool = True) -> Tensor:
        """Create the training labels."""
        # Labels are input_ids
        labels = tokens.clone()
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        # Mask inputs
        if mask_inputs is True:
            input_mask = self._get_input_mask(tokens)
            labels[input_mask] = -100
        return labels
