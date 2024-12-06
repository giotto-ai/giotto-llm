import copy
import gc
import json
import logging
import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Any, Literal, Type

import numpy as np
import psutil
import torch
import transformers
from pydantic import BaseModel, Field, ValidationError, validator
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.utils import ModelOutput

from .consts import DATA_CONFIG_FILENAME, GRID_FORMATTER_CONFIG_FILENAME, MAX_GRID_SIZE
from .data import Dataset
from .prompts.consts import TYPES_OF_PROMPTS
from .prompts.grid_formatter import GridFormatter
from .transforms import Transforms, _BackTransformTestOutput, backtransform_test_output
from .type_aliases import Attempts, Grid, JSONTask, OAIMessage
from .utils import RepeatSampler, write_json


class EvaluationConfig(BaseModel):
    """Config for customizing evaluation behavior.

    Args:
        n_attempts: The number of attempts to return. If `None`, returns all attempts.
        n_transforms: The number of predictions per task with a different transform. Requires the use of random transforms.
        rigid_transforms_all: If `True`, applies all 8 rigid transforms to all tasks and then performs `n_transforms` using other transformations like color permutation, example re-ordering, etc.  Number of transoformed tasks will be 8 * n_transforms
        batch_size: The batch size. Different batch sizes can affect the result due to numerical stability.
        n_dataloader_workers: The number of worker subprocesses used by the DataLoader.
        image_resize_factor: The upscaling factor of images in multi-modal models.
        input_tokens_limit: Limits the number of tokens in the training examples and test input by subsampling the training examples.
        save_generation_metadata: If `True`, saves additional metadata in './generation_metadata'.
        generation_config: Parameters passed to the model.generate method. Typically used to set generation/sampling strategy via `num_beams`, `temperature`, etc. `num_return_sequences=n` will generate `n` responses per input.
    """
    n_attempts: int | None = Field(ge=1, default=None)
    n_transforms: int = Field(ge=1, default=1)
    rigid_transforms_all: bool = False
    batch_size: int = Field(ge=1, default=1)
    n_dataloader_workers: int = Field(ge=1, default=psutil.cpu_count(logical=False))  # type: ignore
    image_resize_factor: int = 3
    input_tokens_limit: int | None = None
    save_generation_metadata: bool = False
    generation_config: dict[str, Any] = {"max_new_tokens": 1024, "num_return_sequences": 1}
    dfs_sampling: bool = False
    dfs_config: dict[str, Any] = {"max_new_tokens": 1024, "threshold": 0.1, "batch_size": 6}

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
                rigid=not config.rigid_transforms_all,
                max_tokens=config.input_tokens_limit,
            )
        prompt_type = self.data_config["prompt_type"]
        assert isinstance(prompt_type, str)
        messages_fn = TYPES_OF_PROMPTS[prompt_type](grid_formatter=self.grid_formatter)
        task_grids: dict[str, list[Grid]] = defaultdict(list)
        task_log_likelihoods: dict[str, list[float]] = defaultdict(list)
        dataset = Dataset(
            tasks=tasks,
            messages_fn=messages_fn,
            model_type=self.model_type,
            transforms=transforms,
            image_resize_factor=config.image_resize_factor,
            rigid_transforms_all=config.rigid_transforms_all
        )
        print(f"RIGID_TRANSFORMS_ALL: {config.rigid_transforms_all},  DATASET LENGTH: {len(dataset)}")
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
            batch_size = len(batch_indices)
            task_ids = [dataset.keys[task_index] for task_index in batch_indices]
            logger.info(f"Analyzing tasks {task_ids=}")

            backtransforms = batch["backtransforms"]
            batch_inputs = batch["batch_inputs"]

            responses = []
    
            if config.dfs_sampling:
                try:
                    responses, log_likelihoods = self.batched_bfs_sampling(batch_inputs["input_ids"], batch_inputs["attention_mask"], 
                                                                           config.dfs_config, logger)

                    batch_indices = [batch_indices[i] for i in range(batch_size) for _ in range(len(responses[i]))]
                    backtransforms = [backtransforms[i] for i in range(batch_size) for _ in range(len(responses[i]))]
                    responses = [response for i in range(batch_size) for response in responses[i]]
                    log_likelihoods = torch.tensor([log_likelihood for i in range(batch_size) for log_likelihood in log_likelihoods[i]])

                    logger.info(f"DFS Sampling completed. Number of responses: {len(responses)}")
                    logger.info("EXAMPLE OF RESPONSES and LOG LIKELIHOODS")
                    logger.info(log_likelihoods[0])
                    logger.info(responses[0])

                except Exception as e:
                    logger.error(f"DFS Sampling failed: {e}")
                    logger.error("Running Greedy Search instead.")
                    batch_indices = batch["batch_indices"]
                    backtransforms = batch["backtransforms"]
                    responses = []

            if  not responses:
                if config.dfs_sampling:
                    logger.info("DFS Sampling pruned all sequences. Running Greed Search.")
                for key in batch_inputs:
                    batch_inputs[key] = batch_inputs[key].to(device=self.model.device)
                output = self._generate(batch_inputs, generation_config)
                input_size = batch_inputs["input_ids"].shape[1]
                responses = self._decode(output.sequences, input_size=input_size)
                log_likelihoods = self._get_log_likelihoods(output, input_size=input_size)
                
                if generation_config.num_return_sequences > 1:
                    batch_indices = [
                        i for i in batch_indices for _ in range(generation_config.num_return_sequences)
                    ]
                    backtransforms = [
                        t for t in backtransforms for _ in range(generation_config.num_return_sequences)
                    ]
                
            del batch_inputs
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

            if config.save_generation_metadata is True:
                self._save_generation_metadata(
                    task_ids=[dataset.keys[task_index] for task_index in batch_indices],
                    output=output,
                    transformed_tasks=batch["transformed_tasks"],
                    responses=responses,
                    attempts=attempts,
                    backtransforms=backtransforms,
                    input_size=input_size,
                )

            for attempt, log_likelihood, idx, backtransform in zip(
                attempts, log_likelihoods, batch_indices, backtransforms, strict=True
            ):
                if attempt is None:
                    continue
                task_id = dataset.keys[idx]
                if config.rigid_transforms_all:
                    task_id = task_id.split("@")[0]
                task_grids[task_id].append(
                    backtransform_test_output(grid=attempt, backtransform=backtransform)
                )
                task_log_likelihoods[task_id].append(log_likelihood.item())

        results: dict[str, Attempts] = self._create_results(
            task_attempts=task_grids,
            task_log_likelihoods=task_log_likelihoods,
            n_attempts=config.n_attempts,
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
    ) -> ModelOutput:
        return self.model.generate(
            **batch_inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
        )

    @torch.no_grad()
    def dfs_sampling(
        self, 
        input_ids: torch.Tensor, 
        dfs_config: dict,
        logger: logging.Logger
    ) -> list[list[tuple[float, list[int]]]]:
        """
        Generate sequences for a batch using DFS sampling with pruning.

        Args:
            dfs_config: Dictionary containing DFS sampling configuration.

        Returns:
            List of valid sequences with their scores for each batch item.
        """
        self.eos_id = self.tokenizer.eos_token_id
        self.threshold = np.log(dfs_config["threshold"]) # Convert probability to log-prob space

        logger.info(f"DFS Sampling with threshold={dfs_config['threshold']}")

        batch_size = input_ids.size(0)
        sequences = [[] for _ in range(batch_size)]
        scores = [[] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            input_prompt = input_ids[batch_idx].tolist()
            logger.info(f"Generating sequences for batch item {batch_idx} and prompt length {len(input_prompt)}")
            self.max_len = len(input_prompt) + dfs_config["max_new_tokens"] # Maximum length of the sequence
            
            output = self._explore(
                tokens=input_prompt,
                score=0,
                cache=None,
                logger=logger
            )
            for seq, score in output:
                scores[batch_idx].append(score)
                sequences[batch_idx].append(self.tokenizer.decode(seq[len(input_prompt):], skip_special_tokens=False))
        
        return sequences, scores


    @torch.no_grad()
    def _explore(
        self,
        tokens: list[int],
        score: float,
        cache: dict | None = None,
        logger: logging.Logger | None = None
    ) -> list[tuple[float, list[int]]]:
        """
        Recursive DFS exploration for a single sequence with pre-filtered valid tokens.

        Args:
            tokens: Current sequence of token IDs.
            score: Cumulative negative log probability.
            cache: Key-value cache for the current sequence.
        Returns:
            List of tuples (sequence, log_probability of the output) for valid continuations.
        """
        # Stop condition
        if tokens[-1] == self.eos_id or len(tokens) >= self.max_len:
            logger.info(f"Stopping at token {len(tokens)} with score {score}, EOS={tokens[-1] == self.eos_id}, max_len={len(tokens) >= self.max_len}")
            return [(tokens.copy(), score)]  # Copy tokens for immutability
        
        # Move cache to GPU when needed
        cache = tuple(tuple(tensor.to(self.model.device) for tensor in layer_cache) for layer_cache in cache) if cache else None
        input_ids = torch.tensor([[tokens[-1]]] if cache else [tokens], dtype=torch.long, device=self.model.device)

        output = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        logits, cache = output.logits, output.past_key_values # logits: (batch_size=1, sequence_length, vocab_size)
        next_token_logits = logits[0, -1, :]  # Logits for the last token
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

        # Move cache to CPU to free GPU memory
        cache = tuple(tuple(tensor.to("cpu") for tensor in layer_cache) for layer_cache in cache)

        del input_ids
        del output, logits
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        valid_sequences = []
        valid_next_tokens = torch.where(score + next_token_log_probs >= self.threshold)[0]

        if len(valid_next_tokens) == 0:
            logger.info(f"Pruning at token {len(tokens)} with score {score}")

        for next_token_id in valid_next_tokens.tolist():
            next_score = score + next_token_log_probs[next_token_id].item() 
            tokens.append(next_token_id)  # Modify tokens in place
            continuations = self._explore(tokens, next_score, cache, logger)
            valid_sequences.extend(continuations)
            tokens.pop()  # Backtrack

        del cache
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return valid_sequences

    def move_cache_to_device(self, cache, device):
        return tuple(tuple(tensor.to(device) for tensor in layer_cache) for layer_cache in cache) if cache else None
    
    def pack_list_of_caches(self, caches):
        # Input: List of caches each of which is a (tuple of layers, each layer is a tuple of tensors)
        # Output: Tuple of layers, each layer is a tuple of tensors, each tensor is a batched
        if not caches:
            return None
        
        output_cache = []
        for layer_cache in zip(*caches):
            output_cache.append(tuple(torch.cat(tesnors, dim=0) for tesnors in zip(*layer_cache)))
        return tuple(output_cache)

    def unpack_cache(self, cache, device="cpu"):
        if not cache:
            return None
        batch_size = len(cache[0][0])
        return [tuple(tuple(tensor[i].unsqueeze(0).to(device) for tensor in layer_cache) for layer_cache in cache) for i in range(batch_size)]

    @torch.no_grad()
    def bfs_sampling(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dfs_config: dict,
        logger: logging.Logger
    ) -> tuple[list[list[str]], list[list[float]]]:
        """
        Generate sequences for a batch using BFS sampling with pruning and batch forward pass.

        Args:
            input_ids: Tensor containing the input prompts for the batch.
            dfs_config: Dictionary containing BFS sampling configuration.
            logger: Logger for debug information.

        Returns:
            A tuple of sequences and their log-probabilities for each batch item.
        """
        eos_id = self.tokenizer.eos_token_id
        threshold = np.log(dfs_config["threshold"])  # Convert probability to log-prob space

        logger.info(f"BFS Sampling with threshold={dfs_config['threshold']}")

        batch_size = input_ids.size(0)
        sequences, scores = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            input_attention_mask = attention_mask[batch_idx].to(self.model.device)
            input_prompt = input_ids[batch_idx].tolist()
            max_len = len(input_prompt) + dfs_config["max_new_tokens"]  # Maximum length of the sequence
            logger.info(f"Generating sequences for batch item {batch_idx} with prompt length {len(input_prompt)}")

            completed_sequences = []

            # Initialize BFS state
            current_input_ids = [input_prompt]
            current_caches = None
            current_sequences = [(input_prompt, 0)]  # List of (sequence, score, cache)

            while current_sequences:
                assert len(current_sequences) < 1 + 1 / dfs_config["threshold"], "Theoretical limit exceeded. Something Wrong with implementation."
                # Prepare input IDs and caches for the current batch

                current_input_ids = torch.tensor(current_input_ids, dtype=torch.long, device=self.model.device)
                current_caches =  self.move_cache_to_device(self.pack_list_of_caches(current_caches), self.model.device)

                # Forward pass
                output = self.model(input_ids=current_input_ids, attention_mask=input_attention_mask, past_key_values=current_caches, use_cache=True)
                logits, current_caches = output.logits, output.past_key_values
                next_token_logits = logits[:, -1, :]
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Unpack caches for the next round
                current_caches = self.unpack_cache(current_caches, device="cpu")

                next_input_ids = []
                next_caches = []
                new_sequences = []
            

                for i, (sequence, score) in enumerate(current_sequences):
                    # Compute valid next tokens
                    valid_next_tokens = torch.where(score + next_token_log_probs[i] >= threshold)[0]

                    if len(valid_next_tokens) == 0:
                        logger.debug(f"Pruning at token {len(sequence)} with score {score}")
                        continue

                    # Expand valid nodes
                    for next_token_id in valid_next_tokens.tolist():
                        next_score = score + next_token_log_probs[i, next_token_id].item()
                        next_sequence = sequence + [next_token_id]

                        # Stop condition
                        if next_token_id == eos_id or len(next_sequence) >= max_len:
                            completed_sequences.append((next_sequence, next_score))
                        else:
                            next_input_ids.append([next_token_id])
                            next_caches.append(current_caches[i])
                            new_sequences.append((next_sequence, next_score))

                current_input_ids = next_input_ids
                current_caches = next_caches
                current_sequences = new_sequences

                del output, logits, next_token_logits, next_token_log_probs
                del next_input_ids, next_caches, new_sequences
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


            # Collect completed sequences
            for seq, score in completed_sequences:
                scores[batch_idx].append(score)
                sequences[batch_idx].append(self.tokenizer.decode(seq[len(input_prompt):], skip_special_tokens=False))
            
            del input_attention_mask
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return sequences, scores
    

    @torch.no_grad()
    def batched_bfs_sampling(
        self,
        batch_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dfs_config: dict,
        logger: logging.Logger
    ) -> tuple[list[list[str]], list[list[float]]]:
        """
        Generate sequences for a batch using BFS sampling with pruning and batch forward pass.

        Args:
            input_ids: Tensor containing the input prompts for the batch.
            dfs_config: Dictionary containing BFS sampling configuration.
            logger: Logger for debug information.

        Returns:
            A tuple of sequences and their log-probabilities for each batch item.
        """        
        eos_id = self.tokenizer.eos_token_id
        threshold = np.log(dfs_config["threshold"])  # Convert probability to log-prob space
        batch_size = dfs_config["batch_size"]

        logger.info(f"BFS Sampling with threshold={dfs_config['threshold']} and batch size={batch_size}")

        prompt_lengths = [len(batch_input_ids[i]) for i in range(len(batch_input_ids))]

        # To return at least one sequence let's generate greedy sequence as well.
        #batch_greedy_sequence = [(batch_input_ids[i].tolist(), 0, None) for i in range(len(batch_input_ids))]  # List of (sequence, score, cache)
        
        completed_sequences = []
        bfs_layer = 0 # BFS layer := number of tokens generated so far

        # Initialize BFS state
        current_input_ids = batch_input_ids
        current_caches = None
        current_sequences = [(batch_input_ids[i].tolist(), 0, i) for i in range(len(batch_input_ids))]  # List of (sequence, score, batch_idx)

        while current_sequences:
            next_input_ids = []
            next_caches = []
            next_sequences = []
            
            for batch_start in range(0, len(current_sequences), batch_size):
                input_ids_batch = current_input_ids[batch_start:batch_start+batch_size]
                attention_mask_batch = attention_mask[[batch_idx for (_, _, batch_idx) in current_sequences[batch_start:batch_start+batch_size]]]
                caches_batch = current_caches[batch_start:batch_start+batch_size] if current_caches else None

                input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long, device=self.model.device)
                # expand attention mask to batch_size x seq_len by adding 1s for new tokens
                attention_mask_batch = torch.cat([attention_mask_batch, torch.ones((len(attention_mask_batch), bfs_layer))], dim=1).to(self.model.device)
                caches_batch = self.move_cache_to_device(self.pack_list_of_caches(caches_batch), self.model.device)
        
                output = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, past_key_values=caches_batch, use_cache=True) 
                logits, caches_batch = output.logits, output.past_key_values # batch_size x seq_len x vocab_size
                next_token_logits = logits[:, -1, :]  # batch_size x vocab_size
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1) # batch_size x vocab_size

                # Unpack caches for the next round
                caches_batch = self.unpack_cache(caches_batch, device="cpu")

                for i, (sequence, score, batch_idx) in enumerate(current_sequences[batch_start:batch_start+batch_size]):
                    # #Continue greedy search
                    # greedy_token = torch.argmax(next_token_log_probs[i]).item()
                    # greedy_score = score + next_token_log_probs[i, greedy_token].item()
                    # logger.info(f"Greedy search: {greedy_token} {greedy_score}")
                    # if greedy_score >= batch_greedy_sequence[batch_idx][1]:
                    #     batch_greedy_sequence[batch_idx] = (sequence + [greedy_token], greedy_score, caches_batch[i])
                    # Compute valid next tokens
                    valid_next_tokens = torch.where(score + next_token_log_probs[i] >= threshold)[0].tolist()
    
                    if len(valid_next_tokens) == 0:
                        logger.info(f"Pruning at token {len(sequence)} with score {score}")
                        continue

                    # Expand valid nodes
                    for next_token_id in valid_next_tokens:
                        next_score = score + next_token_log_probs[i, next_token_id].item()
                        next_sequence = sequence + [next_token_id]

                        # Stop condition
                        if next_token_id == eos_id or len(next_sequence) >= prompt_lengths[batch_idx] + dfs_config["max_new_tokens"]:
                            completed_sequences.append((next_sequence, next_score, batch_idx))
                        else:
                            next_input_ids.append([next_token_id])
                            next_caches.append(caches_batch[i])
                            next_sequences.append((next_sequence, next_score, batch_idx))
                        
                    
                
                del output, logits, next_token_logits, next_token_log_probs
                del input_ids_batch, caches_batch
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # for i in range(len(batch_greedy_sequence)):
            #     seq, score, cache = batch_greedy_sequence[i]
            #     if seq is None:
            #         continue
            #     if seq[-1] == eos_id or len(seq) >= prompt_lengths[i] + dfs_config["max_new_tokens"]:
            #         completed_sequences.append((seq, score, i))
            #         batch_greedy_sequence[i] = (None, np.inf, None)
            #     else:
            #         next_input_ids.append([seq[-1]])
            #         next_caches.append(cache)
            #         next_sequences.append((seq, score, i))

            current_input_ids = next_input_ids
            current_caches = next_caches
            current_sequences = next_sequences

            bfs_layer += 1

            del next_input_ids, next_caches, next_sequences

        # Collect completed sequences
        sequences, scores = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
        for (seq, score, batch_idx) in completed_sequences:
            scores[batch_idx].append(score)
            sequences[batch_idx].append(self.tokenizer.decode(seq[prompt_lengths[batch_idx]:], skip_special_tokens=False))

        return sequences, scores


    def _set_output_token_ids(self) -> None:
        self.output_token_ids: dict[str | int, int] = {
            "start_output": self.tokenizer.vocab[self.grid_formatter.sO_token],
            "end_output": self.tokenizer.vocab[self.grid_formatter.eO_token],
            "start_row": self.tokenizer.vocab[self.grid_formatter.sR_token],
            "end_row": self.tokenizer.vocab[self.grid_formatter.eR_token],
            "eos": self.tokenizer.eos_token_id,
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

    def _create_results(
        self,
        task_attempts: dict[str, list[Grid]],
        task_log_likelihoods: dict[str, list[float]],
        n_attempts: int | None,
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
            attempts = task_attempts[split_task_id]
            # There can be duplicate attempts, so sum the log likelihood of duplicates
            attempt_log_likelihoods: dict[str, list[float]] = defaultdict(list)
            for i, attempt in enumerate(attempts):
                attempt_log_likelihoods[str(attempt)].append(task_log_likelihoods[split_task_id][i])
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
        generated_tokens = output.sequences[:, input_size:]
        # Stack logits to get shape [batch_size, sequence_length, vocab_size]
        logits = torch.stack(output.scores, dim=1)
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
