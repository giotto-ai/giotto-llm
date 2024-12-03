import copy
import json
import logging
import os
import pathlib
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch

from giotto_llm.causal_lm.models import CausalLMWrapper
from giotto_llm.consts import DEFAULT_ATTEMPT, ROOT_PATH
from giotto_llm.logs import get_named_logger
from giotto_llm.multimodal.molmo import MolmoWrapper
from giotto_llm.multimodal.qwen import QwenVLWrapper
from giotto_llm.reader import ReaderMany
from giotto_llm.transforms import Transforms
from giotto_llm.validation.args import parse_arguments
from giotto_llm.wrapper import EvaluationConfig

HOME = pathlib.Path().home()

WRAPPER_CLS_TYPES = {
    "CausalLM": CausalLMWrapper,
    "QwenVL": QwenVLWrapper,
    "Molmo": MolmoWrapper,
}


def main(
    model_id: str,
    model_name: str,
    wrapper_cls: Any,
    cpu_only: bool,
    quantization: str,
    evaluation_config: EvaluationConfig,
    dataset_dir: str,
    dataset_type: str,
    start_index_tasks: int,
    end_index_tasks: int,
    gpu_index: int,
    logger: logging.Logger,
) -> None:
    """Create submission for a subset of tasks, defined by a range of indices."""
    today = datetime.now()
    date_string = today.strftime("%Y_%m_%d")
    logger.info(
        f">>> Running {model_name=} validation with {evaluation_config=} in range [{start_index_tasks}, {end_index_tasks})"
    )

    logger.info(f">>> Using only GPU at index {gpu_index}")

    _safe_tasks = ReaderMany(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        read_test_output=False,
    ).read_tasks()
    # Note: We create "safe" tasks to be sure that we do not pass
    # the test output during inference
    safe_tasks_ids = sorted(_safe_tasks)
    safe_tasks_ids = safe_tasks_ids[
        start_index_tasks:end_index_tasks
    ]  # Note: we do include `end_index_tasks`
    safe_tasks: dict = {task_id: _safe_tasks[task_id] for task_id in safe_tasks_ids}

    logger.info(f"Creating an instance of {wrapper_cls=} with {quantization=} on {gpu_index=}")
    wrapper = wrapper_cls(model_id=model_id, gpu_index=gpu_index, quantization=quantization)
    if cpu_only:
        wrapper.model.to("cpu")

    results = wrapper.evaluate(
        tasks=safe_tasks,
        logger=logger,
        config=evaluation_config,
    )

    tasks_with_output = ReaderMany(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        read_test_output=True,
    ).read_tasks()

    default_attempts = {"attempt_1": DEFAULT_ATTEMPT, "attempt_2": DEFAULT_ATTEMPT}
    submission: dict = {
        task_id: [
            copy.deepcopy(default_attempts) for _ in range(len(tasks_with_output[task_id]["test"]))
        ]
        for task_id in safe_tasks_ids
    }
    count_solved = 0
    total = 0
    for task_id, attempts in results.items():
        attempts_task_id = []
        for idx_i in range(len(safe_tasks[task_id]["test"])):
            if idx_i not in attempts:
                attempts_task_id.append(
                    {"attempt_1": DEFAULT_ATTEMPT, "attempt_2": DEFAULT_ATTEMPT}
                )
            logger.info(f">>> Evaluating test pair at index {idx_i=} for {task_id=}")
            grids = attempts[idx_i]
            expected_grid = tasks_with_output[task_id]["test"][idx_i]["output"]

            logger.info(f">>> Grids\n{grids=}\n{expected_grid=}")
            logger.info("---")

            for grid in grids:
                if grid == expected_grid:
                    count_solved += 1
                    break
            total += 1
            logger.info(f">>> Currently {count_solved=}/{total}")

            attempts_task_id.append(
                {
                    "attempt_1": grids[0] if len(grids) > 0 else DEFAULT_ATTEMPT,
                    "attempt_2": grids[1] if len(grids) > 1 else DEFAULT_ATTEMPT,
                }
            )

        submission[task_id] = attempts_task_id

    logger.info(f">>> Final {count_solved=}/{total}")

    submission_dir_path = str(ROOT_PATH / "subs" / date_string)

    submission_file_path = os.path.join(
        submission_dir_path,
        f"submission_{model_name}_{start_index_tasks:03d}_{end_index_tasks:03d}_gpu_{gpu_index}.json",
    )
    os.makedirs(submission_dir_path, exist_ok=True)
    with open(submission_file_path, "w") as f:
        json.dump(submission, f)

    logger.info(f">>> Finished {submission_file_path}")


if __name__ == "__main__":
    arguments = parse_arguments()

    random.seed(arguments.random_seed)
    torch.manual_seed(arguments.random_seed)
    np.random.seed(arguments.random_seed)

    model_name = pathlib.Path(arguments.finetuned_model_id).parts[-1]
    logger = get_named_logger(
        name=f"validation_{model_name}_dataset_{arguments.dataset_type}",
        log_level=logging.INFO,
        enable_log_to_file=True,
        project_root=str(ROOT_PATH),
        output_dir="logs",
    )

    evaluation_config = EvaluationConfig(
        n_attempts=arguments.n_attempts,
        n_transforms=arguments.n_transforms,
        batch_size=arguments.batch_size,
        n_dataloader_workers=arguments.n_dataloader_workers,
        input_tokens_limit=arguments.input_tokens_limit,
        save_generation_metadata=arguments.save_generation_metadata,
        generation_config={
            "max_new_tokens": arguments.max_new_tokens,
            "num_return_sequences": arguments.num_return_sequences,
            "num_beams": arguments.num_beams,
        },
    )

    main(
        model_id=arguments.finetuned_model_id,
        model_name=model_name,
        wrapper_cls=WRAPPER_CLS_TYPES[arguments.wrapper_cls_type],
        cpu_only=arguments.cpu_only,
        quantization=arguments.quantization,
        evaluation_config=evaluation_config,
        dataset_dir=arguments.dataset_dir,
        dataset_type=arguments.dataset_type,
        start_index_tasks=arguments.start_index_tasks,
        end_index_tasks=arguments.end_index_tasks,
        gpu_index=arguments.gpu_index,
        logger=logger,
    )
