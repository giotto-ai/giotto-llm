import argparse
import pathlib

import psutil
from easydict import EasyDict

from giotto_llm.causal_lm.models import CausalLMWrapper
from giotto_llm.consts import ROOT_PATH
from giotto_llm.prompts.consts import TYPES_OF_PROMPTS

HOME = pathlib.Path().home()


def parse_arguments() -> EasyDict:
    parser = argparse.ArgumentParser(
        description=f"Run fine-tuned LLMs on evaluation or training data."
    )

    parser.add_argument(
        "--finetuned_model_id",
        type=str,
        required=True,
        help="Directory containing merged fine-tuned model",
    )

    default_data_dir = f"{ROOT_PATH}/kaggle/input"
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=default_data_dir,
        help="Path to directory containing data",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="full",  # Note: all 800 tasks
        help="One of [training, evaluation, test, full]",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size at inference time",
    )

    parser.add_argument(
        "--wrapper_cls_type",
        type=str,
        default="CausalLM",
        help="Wrapper class of LLM transformer models",
    )

    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Flag to run inference on CPU",
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization to use for adapters",
    )

    parser.add_argument(
        "--n_dataloader_workers",
        type=int,
        default=psutil.cpu_count(logical=False),  # Note: number of physical cores
        help="Number of worker processes used by DataLoader",
    )

    parser.add_argument(
        "--n_attempts",
        type=int,
        required=True,
        help="Maximum number of attempts to return for each ARC task",
    )

    parser.add_argument(
        "--n_transforms",
        type=int,
        required=True,
        help="Number of random augmentation to apply",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,  # Should be large enough to contain any output grid
        help="Number of random augmentation to apply",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=2,
        help="Number of encoded grids to return",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        required=True,
        help="Number of beams to use for beam search decoding",
    )

    parser.add_argument(
        "--start_index_tasks",
        type=int,
        default=0,
        help="Starting index on sorted task ids",
    )

    parser.add_argument(
        "--end_index_tasks",
        type=int,
        default=0,
        help="Ending index on sorted task ids",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="Index of the GPU device to use",
    )

    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=1.0,
        required=False,
        help="Monitoring interval for GPU, CPU and RAM usage.",
    )

    parser.add_argument(
        "--num_tasks_per_gpu_process",
        type=int,
        default=0,
        help="Number of tasks processed by each child subprocess. This ensures that if a batch of task fails, we still continue execution on the remaining batches. If batch size is set to 1, each child subprocess in parallelize will process one task at a time. If batch is set to 0, the length of the batches is automatically inferred from the total number of tasks and the number of available GPUs.",
    )

    parser.add_argument(
        "--input_tokens_limit",
        type=int,
        default=None,
        help="limit the number of tokens of the training examples and test input, by subsampling the training examples.",
    )

    parser.add_argument(
        "--save_generation_metadata",
        action="store_true",
        help="Save additional metadata of generated attempts in ./generation_metadata",
    )

    arguments = parser.parse_args()

    arguments = EasyDict(
        {
            "finetuned_model_id": arguments.finetuned_model_id,
            "dataset_dir": arguments.dataset_dir,
            "dataset_type": arguments.dataset_type,
            "n_dataloader_workers": arguments.n_dataloader_workers,
            "batch_size": arguments.batch_size,
            "wrapper_cls_type": arguments.wrapper_cls_type,
            "cpu_only": arguments.cpu_only,
            "quantization": arguments.quantization,
            "n_attempts": arguments.n_attempts,
            "n_transforms": arguments.n_transforms,
            "max_new_tokens": arguments.max_new_tokens,
            "num_return_sequences": arguments.num_return_sequences,
            "num_beams": arguments.num_beams,
            "start_index_tasks": arguments.start_index_tasks,
            "end_index_tasks": arguments.end_index_tasks,
            "random_seed": arguments.random_seed,
            "gpu_index": arguments.gpu_index,
            "monitor_interval": arguments.monitor_interval,
            "num_tasks_per_gpu_process": arguments.num_tasks_per_gpu_process,
            "input_tokens_limit": arguments.input_tokens_limit,
            "save_generation_metadata": arguments.save_generation_metadata,
        }
    )
    return arguments
