import argparse
import logging
from typing import Any, Literal

from easydict import EasyDict

from ..consts import ROOT_PATH
from ..logs import get_named_logger

map_log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def parse_arguments_main() -> EasyDict:
    """Get the parameters required to start fine-tuning."""

    parser = argparse.ArgumentParser(description=f"Fine-tuning")

    parser.add_argument(
        "--kaggle_mode",
        action="store_true",
        help="Run the kaggle mode solution",
    )

    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        help="Use unsloth for fine-tuning and inference",
    )

    default_data_dir = f"{ROOT_PATH}/kaggle/input"
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=default_data_dir,
        help="Path to directory containing data",
    )

    parser.add_argument(
        "--dataset_category",
        type=str,
        default="full",  # Note: all 800 tasks
        help="One of [training, evaluation, test, full]",
    )

    parser.add_argument(
        "--num_tasks_per_gpu_process",
        type=int,
        default=1,
        help="Number of tasks processed by each child subprocess. This ensures that if a batch of task fails, we still continue execution on the remaining batches. If batch size is set to 1, each child subprocess in parallelize will process one task at a time. If batch is set to 0, the length of the batches is automatically inferred from the total number of tasks and the number of available GPUs.",
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
        "--gpu_index",
        type=int,
        default=0,
        help="Index of the GPU device to use",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Huggingface base-model id.",
    )

    parser.add_argument(
        "--wrapper",
        type=str,
        required=True,
        help="Which wrapper to use for the model.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output dir to save model to.",
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Which quantization setting to use,",
    )

    parser.add_argument(
        "--transform_background_color",
        type=bool,
        default=True,
        help="Whether or not to transform background color in the dataloader,",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="The initial learning rate.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size on each device.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="The eval batch size on each device.",
    )

    parser.add_argument(
        "--eval_n_attempts",
        type=int,
        default=2,
        help="The eval number of attempts.",
    )

    parser.add_argument(
        "--eval_n_transforms",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--eval_rigid_transforms_all",
        action="store_true",
        help="If used, applies all 8 rigid transforms to all tasks and then performs `n_transforms` using other transformations like color permutation, example re-ordering, etc. Number of transoformed tasks will be 8 * n_transforms.",
    )

    parser.add_argument(
        "--eval_num_return_sequences",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--eval_num_beams",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="The number of gradient accumulation steps.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="The number of training epochs",
    )

    parser.add_argument(
        "--neftune_noise_alpha",
        type=float,
        default=None,
        help="The amount of noise to add embeddings",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="prompt_solve_short",
        help="The type of text prompt to use",
    )

    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        nargs="+",
        help="The target modules for Lora. If not specified, target all linear layers.",
    )

    parser.add_argument(
        "--logging_steps",
        type=str,
        default=None,
        help="Number of steps between each log entry. If not specified, log per epoch.",
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1,
        help="Number of steps between each pass of the evaluation set. If not specified, evaluate once per epoch.",
    )

    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Enable options that save memory at the expense of performance.",
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.2,
        help="The dropout used in Lora.",
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha used in Lora.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="The r used in Lora.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="The early stopping patience.",
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="How many checkpoints to keep.",
    )

    parser.add_argument(
        "--untie_word_embeddings",
        action="store_true",
        help="Untie the word embeddings from the classification head.",
    )

    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="info",
        help="Logging level",
    )

    parser.add_argument(
        "--logger_name",
        type=str,
        default="finetuning",
        help="Logger name",
    )

    parser.add_argument(
        "--enable_log_to_file",
        action="store_true",
        help="Save runtime information in a log file",
    )

    arguments = parser.parse_args()

    # Validation
    assert arguments.log_level in {"debug", "info", "warning", "error", "critical"}

    log_level = map_log_levels[arguments.log_level]
    logger = get_named_logger(
        name=f"{arguments.logger_name}_{arguments.dataset_dir}",
        log_level=log_level,
        enable_log_to_file=arguments.enable_log_to_file,
        project_root=str(ROOT_PATH),
        output_dir="logs",
    )

    return EasyDict(
        {
            "num_tasks_per_gpu_process": arguments.num_tasks_per_gpu_process,
            "kaggle_mode": arguments.kaggle_mode,
            "use_unsloth": arguments.use_unsloth,
            "logger": logger,
            "dataset_dir": arguments.dataset_dir,
            "dataset_category": arguments.dataset_category,
            "start_index_tasks": arguments.start_index_tasks,
            "end_index_tasks": arguments.end_index_tasks,
            "gpu_index": arguments.gpu_index,
            "model_id": arguments.model_id,
            "wrapper": arguments.wrapper,
            "output_dir": arguments.output_dir,
            "quantization": arguments.quantization,
            "transform_background_color": arguments.transform_background_color,
            "learning_rate": arguments.learning_rate,
            "batch_size": arguments.batch_size,
            "eval_batch_size": arguments.eval_batch_size,
            "eval_n_attempts": arguments.eval_n_attempts,
            "eval_n_transforms": arguments.eval_n_transforms,
            "eval_rigid_transforms_all": arguments.eval_rigid_transforms_all,
            "eval_num_return_sequences": arguments.eval_num_return_sequences,
            "eval_num_beams": arguments.eval_num_beams,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "num_train_epochs": arguments.num_train_epochs,
            "neftune_noise_alpha": arguments.neftune_noise_alpha,
            "prompt_type": arguments.prompt_type,
            "lora_target_modules": arguments.lora_target_modules,
            "logging_steps": arguments.logging_steps,
            "eval_steps": arguments.eval_steps,
            "low_memory": arguments.low_memory,
            "lora_dropout": arguments.lora_dropout,
            "lora_alpha": arguments.lora_alpha,
            "lora_r": arguments.lora_r,
            "early_stopping_patience": arguments.early_stopping_patience,
            "save_total_limit": arguments.save_total_limit,
            "untie_word_embeddings": arguments.untie_word_embeddings,
        }
    )
