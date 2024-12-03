import copy
import glob
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

from giotto_llm.consts import ROOT_PATH
from giotto_llm.logs import get_named_logger
from giotto_llm.online_fine_tuning.args import parse_arguments_main
from giotto_llm.reader import ReaderMany
from giotto_llm.validation.monitor import log_resource_usage

TIME_LIMIT = 1_000_000


process_counter = 0


def merge_jsons(dir_path: str, output_file: str) -> None:
    """Merge all json files in a directory into a single file."""
    all_data = {}
    for file_path in glob.glob(f"{dir_path}/*.json"):
        with open(file_path, "r") as f:
            all_data.update(json.load(f))
    with open(output_file, "w") as f:
        json.dump(all_data, f)


def process_batch(
    arguments: dict,
    model_name: str,
    batch_start: int,
    batch_end: int,
    gpu_index: int,
    logs_dir_path: str,
    logger: logging.Logger,
) -> tuple[subprocess.Popen, int]:
    """
    Start a new process on the gpu_index on tasks from batch_start to batch_end (indices).
    """
    global process_counter

    cmd = assemble_commmand_from_arguments(
        arguments=arguments,
        gpu_index=gpu_index,
        start_index_tasks=batch_start,
        end_index_tasks=batch_end,
    )
    log_file_gpu = os.path.join(
        str(logs_dir_path), f"output_validation_{model_name}_gpu_{gpu_index}.log"
    )
    logger.info(f">>> Going to run the following command as subprocess")
    logger.info(f"\t- {cmd}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if arguments["kaggle_mode"]:
        env["WANDB_MODE"] = "disabled"

    with open(log_file_gpu, "a") as f:
        f.write(f"Processing tasks from {batch_start} to {batch_end} on GPU {gpu_index}")
        process_counter += 1
        pro = subprocess.Popen(
            cmd,
            shell=False,
            start_new_session=True,
            stdout=f,
            stderr=f,
            env=env,  # Pass the environment with CUDA_VISIBLE_DEVICES
        )
    process_id = process_counter
    return pro, process_id


def main(
    arguments: EasyDict,
    logger: logging.Logger,
) -> None:
    TIME_LIMIT = (
        (
            36000
            if not os.environ.get("ONLINE_FINETUNING_TIME_LIMIT")
            else int(os.environ["ONLINE_FINETUNING_TIME_LIMIT"])
        )
        if arguments["kaggle_mode"]
        else 1_000_000
    )
    model_name = Path(arguments.model_id).parts[-1]
    today = datetime.now()
    date_string = today.strftime("%Y_%m_%d")
    submission_dir_path = os.path.join(arguments["output_dir"], "predictions")
    os.makedirs(str(submission_dir_path), exist_ok=True)
    logs_dir_path = f"{arguments['output_dir']}/logs"
    os.makedirs(str(logs_dir_path), exist_ok=True)

    total_start_time = time.time()
    logger.info(f">>> Starting parallelized validation")
    logger.info(f">>> Saving submission.json files to {submission_dir_path}")
    logger.info(f">>> Saving logs to {logs_dir_path}")
    logger.info("-" * 30)

    number_gpus = count_available_gpus()
    logger.info(f">>> Found {number_gpus} GPUs")

    # Read tasks
    tasks = ReaderMany(
        dataset_dir=arguments.dataset_dir,
        dataset_type=arguments.dataset_category,
        read_test_output=False,
    ).read_tasks()
    tasks_ids: list[str] = sorted(tasks)
    tasks_ids = tasks_ids[arguments.start_index_tasks : arguments.end_index_tasks]
    total_num_tasks = len(tasks_ids)
    logger.info(
        f">>> Found {total_num_tasks} tasks with {arguments.dataset_dir=}, {arguments.dataset_category=}"
    )

    # Divide tasks in batches
    batch_size = (
        int(arguments.num_tasks_per_gpu_process)
        if arguments.num_tasks_per_gpu_process > 0
        else total_num_tasks // number_gpus
    )
    total_batches = np.ceil(total_num_tasks / batch_size).astype("int")
    logger.info(
        f">>> Dividing tasks into {total_batches} batches, each with up to {batch_size} tasks"
    )

    def get_batch_start_end_idxs(batch_id: int) -> tuple[int, int]:
        assert batch_id < total_batches, f"{batch_id=} is greater than {total_batches=}"
        start_idx = arguments.start_index_tasks + batch_size * batch_id
        end_idx = min(start_idx + batch_size, arguments.end_index_tasks)
        return start_idx, end_idx

    available_gpus = deque(range(number_gpus))
    remaining_batches = deque(range(total_batches))
    active_processes: dict[int, tuple[subprocess.Popen, int, int]] = (
        {}
    )  # Dictionary to keep track of active child processes and their assigned batches and gpu index

    # Start processes
    while available_gpus:
        gpu_idx = available_gpus.popleft()  # Get next available GPU
        if remaining_batches:
            next_batch = remaining_batches.popleft()  # Get next available batch
            start_idx, end_idx = get_batch_start_end_idxs(next_batch)
            pro, pro_id = process_batch(
                arguments, model_name, start_idx, end_idx, gpu_idx, logs_dir_path, logger
            )  # Start child process
            active_processes[pro_id] = (pro, next_batch, gpu_idx)

    logger.info(f">>> Started subprocesses: {active_processes}")
    time.sleep(20)

    finished_batches: list[int] = []
    failed_batches: list[int] = []
    finished_task_ids: list[int] = []
    failed_task_ids: list[str] = []
    while active_processes:
        try:
            elapsed_time = time.time() - total_start_time
            if elapsed_time > TIME_LIMIT:
                logger.warning(
                    f"TIME_LIMIT of {TIME_LIMIT} seconds exceeded. Terminating all processes."
                )
                for pro_id, (pro, batch_id, gpu_idx) in list(active_processes.items()):
                    pro.terminate()  # Attempt to terminate gracefully
                    time.sleep(1)  # Give it a second to close properly
                    if pro.poll() is None:  # Check if process is still running
                        logger.warning(f"Process {pro_id} did not terminate. Killing it.")
                        pro.kill()  # Forcefully kill the process if still running
                break

            finished_processes: list[int] = []
            for pro_id, (pro, batch_id, gpu_idx) in list(active_processes.items()):
                if pro.poll() is not None:  # Process finished
                    finished_processes.append(pro_id)
                    available_gpus.append(gpu_idx)  # Release GPU
                    start_idx, end_idx = get_batch_start_end_idxs(batch_id)

                    if pro.returncode == 0:  # Process finished successfully
                        finished_batches.append(batch_id)
                        logger.info(
                            f">>> Task batch {batch_id} successful, with tasks {tasks_ids[start_idx:end_idx]}"
                        )
                    else:  # Process failed
                        failed_batches.append(batch_id)
                        logger.warning(
                            f">>> Task batch {batch_id} failed, with tasks {tasks_ids[start_idx:end_idx]}"
                        )

                    if remaining_batches:
                        next_batch = remaining_batches.popleft()  # Get next available batch
                        start_idx, end_idx = get_batch_start_end_idxs(next_batch)
                        pro, pro_id = process_batch(
                            arguments,
                            model_name,
                            start_idx,
                            end_idx,
                            gpu_idx,
                            logs_dir_path,
                            logger,
                        )  # Start child process
                        active_processes[pro_id] = (pro, next_batch, gpu_idx)
                        time.sleep(20)

            for pro_id in finished_processes:
                del active_processes[pro_id]
        except Exception as e:
            logger.warning(f"Exception caught, error {e}. Terminating all processes.")
            for pro_id, (pro, batch_id, gpu_idx) in list(active_processes.items()):
                pro.terminate()
            break

    total_end_time = time.time()
    logger.info(f">>> Finished after {total_end_time - total_start_time:.2f} seconds")
    logger.info(f">>> Successful batches: {finished_batches}")
    logger.info(f">>> Failed batches: {failed_batches}")
    logger.info(f">>> Tasks with predictions: {finished_task_ids}")
    logger.info(f">>> Tasks without prediction: {failed_task_ids}")

    submission_dir_path = os.path.join(arguments["output_dir"], "predictions")
    failed_tasks_dir = os.path.join(arguments["output_dir"], "failed_tasks")

    os.makedirs(submission_dir_path, exist_ok=True)
    os.makedirs(failed_tasks_dir, exist_ok=True)

    merge_jsons(submission_dir_path, f"{submission_dir_path}/submission.json")
    merge_jsons(failed_tasks_dir, f"{failed_tasks_dir}/failed_tasks.json")


def count_available_gpus() -> int:
    """Return the number of GPUs in this node"""
    if not torch.cuda.is_available():
        raise SystemError("No GPU device is available to torch")

    return torch.cuda.device_count()


def assemble_commmand_from_arguments(
    arguments: EasyDict,
    gpu_index: int,
    start_index_tasks: int,
    end_index_tasks: int,
) -> list[str]:
    """Assemble a command to be run as a subproces that will use exactly one GPU."""
    cmd = [
        "python",
        "-m",
        "giotto_llm.online_fine_tuning",
        "--dataset_dir",
        str(arguments["dataset_dir"]),
        "--dataset_category",
        str(arguments["dataset_category"]),
        "--start_index_tasks",
        str(int(start_index_tasks)),
        "--end_index_tasks",
        str(int(end_index_tasks)),
        "--gpu_index",
        str(gpu_index),
        "--model_id",
        str(arguments["model_id"]),
        "--wrapper",
        str(arguments["wrapper"]),
        "--output_dir",
        str(arguments["output_dir"]),
        "--quantization",
        str(arguments["quantization"]),
        "--transform_background_color",
        str(arguments["transform_background_color"]),
        "--learning_rate",
        str(arguments["learning_rate"]),
        "--batch_size",
        str(arguments["batch_size"]),
        "--gradient_accumulation_steps",
        str(arguments["gradient_accumulation_steps"]),
        "--num_train_epochs",
        str(arguments["num_train_epochs"]),
        "--neftune_noise_alpha",
        str(arguments["neftune_noise_alpha"]),
        "--prompt_type",
        str(arguments["prompt_type"]),
        # "--lora_target_modules",
        # arguments["lora_target_modules"],
        "--eval_steps",
        str(arguments["eval_steps"]),
        "--lora_dropout",
        str(arguments["lora_dropout"]),
        "--lora_alpha",
        str(arguments["lora_alpha"]),
        "--lora_r",
        str(arguments["lora_r"]),
        "--save_total_limit",
        str(arguments["save_total_limit"]),
    ]

    if arguments["kaggle_mode"]:
        cmd.append("--kaggle_mode")

    return cmd


if __name__ == "__main__":
    arguments = parse_arguments_main()
    logger = get_named_logger(
        name=f"parallel_fine_tuning",
        log_level=logging.INFO,
        enable_log_to_file=True,
        project_root=str(ROOT_PATH),
        output_dir=f"{arguments['output_dir']}/logs",
    )

    model_name = Path(arguments.model_id).name
    monitor_csv_path = f"{arguments['output_dir']}/monitor.csv"

    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=log_resource_usage,
        args=(monitor_csv_path, 1.0, logger),
        daemon=True,
    )
    monitor_thread.start()

    try:
        main(arguments, logger=logger)
    finally:
        logger.info(">>> Main execution finished. Monitoring thread will stop.")
