import copy
import json
import logging
import os
import subprocess
from typing import Literal

import numpy as np
import torch
from huggingface_hub._login import _login as hf_login
from peft import LoraConfig
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from giotto_llm.causal_lm.models import CausalLMWrapper
from giotto_llm.consts import DEFAULT_ATTEMPT, ROOT_PATH
from giotto_llm.data import Dataset
from giotto_llm.finetuning.merge import merge_model
from giotto_llm.online_fine_tuning.args import parse_arguments_main
from giotto_llm.online_fine_tuning.utils import MAP_WRAPPER, OnlineFinetuningConfig
from giotto_llm.plot.matplotlib_plots import plot_predictions
from giotto_llm.prompts.consts import TYPES_OF_PROMPTS
from giotto_llm.prompts.grid_formatter import GridFormatter
from giotto_llm.reader import ReaderMany, ReaderOneOnlineFinetuning, ReaderPickle
from giotto_llm.transforms import Transforms, transform_task
from giotto_llm.type_aliases import JSONTask
from giotto_llm.utils import is_tf32_supported, split_tasks_by_test, write_json
from giotto_llm.wrapper import EvaluationConfig

BASE_CONFIG = {
    "wrapper": CausalLMWrapper,
    "wrapper_kwargs": {
        "model_id": "",
        "quantization": "no",
    },
    "evaluation_config": {
        "batch_size": 1,
        "n_attempts": 2,
        "n_transforms": 4,
        "rigid_transforms_all": False,
        "generation_config": {
            "max_new_tokens": 1024,
            "num_return_sequences": 1,
            "num_beams": 1,
        },
    },
}


def get_peft_config(
    target_modules: list[str] | str, dropout: float, alpha: int, r: int
) -> LoraConfig:
    """LoRA config based on QLoRA paper & Sebastian Raschka experiment"""
    return LoraConfig(
        lora_alpha=alpha,
        lora_dropout=dropout,
        r=r,
        bias="none",
        target_modules=target_modules,
        use_rslora=True,
        task_type="CAUSAL_LM",
    )


def get_sft_config(config: OnlineFinetuningConfig) -> SFTConfig:
    """Get the SFTConfig"""
    sft_config = SFTConfig(
        do_eval=not config.kaggle_mode,
        output_dir=config.output_dir,
        logging_dir=f"{config.output_dir}/logs",
        eval_strategy=(
            "no" if config.kaggle_mode else "epoch" if config.eval_steps is None else "steps"
        ),
        eval_steps=config.eval_steps,
        prediction_loss_only=True,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_seq_length=20_000,  # Not used for anything with custom collator
        eval_accumulation_steps=config.gradient_accumulation_steps,
        torch_empty_cache_steps=1 if config.low_memory is True else None,
        fp16_full_eval=config.low_memory,
        learning_rate=config.learning_rate,
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        save_strategy=(
            "no" if config.kaggle_mode else "epoch" if config.eval_steps is None else "steps"
        ),
        save_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,  # Can affect memory use
        save_only_model=True,
        seed=42,
        data_seed=42,
        fp16=not is_tf32_supported(),
        logging_strategy="epoch" if config.logging_steps is None else "steps",
        logging_steps=config.logging_steps,
        bf16=is_tf32_supported(),
        tf32=is_tf32_supported(),
        dataloader_num_workers=4,  # Should be sensible default
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch_fused" if config.quantization is None else "adamw_8bit",
        report_to=None,
        gradient_checkpointing=True,  # Set to False for (possibly) faster training at the expense of memory
        neftune_noise_alpha=config.neftune_noise_alpha,
        use_liger_kernel=True,  # Still runs if not available
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",  # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
        dataloader_pin_memory=not config.low_memory,
    )
    sft_config.remove_unused_columns = False
    return sft_config


def get_train_dataset(  # type: ignore
    config: OnlineFinetuningConfig,
    model_type: Literal["image-text-to-text", "text-to-text"],
    grid_formatter: GridFormatter,
    task_name,
    demo_tasks,
) -> Dataset:
    tasks = ReaderOneOnlineFinetuning(
        task_name, demo_tasks, test_solutions=None, is_test=False
    ).read_tasks()

    transforms = Transforms(
        test=False,
        order="reorder",
        color="all" if config.transform_background_color is True else "foreground",
        limit_colors=model_type == "image-text-to-text",
        rigid=True,
    )
    dataset = Dataset(
        tasks=tasks,
        transforms=transforms,
        messages_fn=TYPES_OF_PROMPTS[config.prompt_type](grid_formatter=grid_formatter),
        model_type=model_type,
    )
    return dataset


def get_eval_dataset(  # type: ignore
    config: OnlineFinetuningConfig,
    model_type: Literal["image-text-to-text", "text-to-text"],
    grid_formatter: GridFormatter,
    task_name,
    demo_tasks,
    test_solutions,
) -> Dataset:
    tasks = ReaderOneOnlineFinetuning(
        task_name, demo_tasks, test_solutions=test_solutions, is_test=True
    ).read_tasks()

    transforms = Transforms(
        test=False,
        order=None,
        color=None,
        limit_colors=model_type == "image-text-to-text",
        rigid=False,
    )

    dataset = Dataset(
        tasks=tasks,
        transforms=transforms,
        messages_fn=TYPES_OF_PROMPTS[config.prompt_type](grid_formatter=grid_formatter),
        model_type=model_type,
    )
    return dataset


def save_eval_results(  # type: ignore
    logger,
    task_name,
    demo_tasks,
    test_solutions,
    model_config,
    submission_save_path,
    image_save_path,
):
    tasks = ReaderOneOnlineFinetuning(
        task_name, demo_tasks, test_solutions=test_solutions, is_test=True
    ).read_tasks()

    wrapper = model_config["wrapper"](**model_config["wrapper_kwargs"])
    logger.info("Starting evaluation")
    results = wrapper.evaluate(
        tasks=tasks,
        logger=logger,
        config=EvaluationConfig(
            **model_config["evaluation_config"],
        ),
    )
    submission: dict = {task_id: [] for task_id in tasks.keys()}
    count_solved = 0
    total = 0
    for index_task, (task_id, attempts) in enumerate(results.items()):
        attempts_task_id = []
        for idx_i in range(len(tasks[task_id]["test"])):
            if idx_i not in attempts:
                attempts_task_id.append(
                    {"attempt_1": DEFAULT_ATTEMPT, "attempt_2": DEFAULT_ATTEMPT}
                )
            else:
                logger.info(f">>> Evaluating {idx_i=} for {task_id=}")
                grids = attempts[idx_i]
                expected_grid = tasks[task_id]["test"][idx_i]["output"]

                # logger.info(f">>> Grids\n{grids}\n{expected_grid}")
                logger.info("---")

                for grid in grids:
                    if grid == expected_grid:
                        count_solved += 1
                        break

                attempts_task_id.append(
                    {
                        "attempt_1": grids[0],
                        "attempt_2": DEFAULT_ATTEMPT if len(grids) == 1 else grids[1],
                    }
                )

            total += 1
            logger.info(f">>> Currently {count_solved=}/{total}")

            plot_predictions(
                tasks[task_id],
                test_id=idx_i,
                predictions=attempts_task_id[-1].values(),
                save_path=f"{image_save_path}_{idx_i}.png",
            )

        submission[task_id] = attempts_task_id

    if count_solved > 0:
        logger.info("\033[95m" + f"TASK IS SOLVED: {count_solved}/{total}" + "\033[0m")
    else:
        logger.info("\033[94m" + f"TASK IS NOT SOLVED: {count_solved}/{total}" + "\033[0m")

    with open(submission_save_path, "w") as f:
        json.dump(submission, f)

    logger.info("Finished")

    return count_solved, total


def run_inference(logger, task_name, demo_tasks, model_config, submission_save_path):  # type: ignore
    wrapper = model_config["wrapper"](**model_config["wrapper_kwargs"])
    results = wrapper.evaluate(
        tasks={task_name: demo_tasks},
        logger=logger,
        config=EvaluationConfig(
            **model_config["evaluation_config"],
        ),
    )

    attempts_task_id = []
    attempts = results[task_name]
    for idx_i in range(len(demo_tasks["test"])):
        if idx_i not in attempts:
            attempts_task_id.append({"attempt_1": DEFAULT_ATTEMPT, "attempt_2": DEFAULT_ATTEMPT})
            raise ValueError(f"Didn't attempted for test {idx_i}")
        else:
            grids = attempts[idx_i]
            attempts_task_id.append(
                {
                    "attempt_1": grids[0],
                    "attempt_2": DEFAULT_ATTEMPT if len(grids) == 1 else grids[1],
                }
            )

    with open(submission_save_path, "w") as f:
        json.dump({task_name: attempts_task_id}, f)


def main(
    logger: logging.Logger,
    base_config: OnlineFinetuningConfig,
    start_index_tasks: int,
    end_index_tasks: int,
    gpu_index: int,
) -> None:
    """Finetune model"""
    logger.info("Creating training dataset")

    image_prediction_dir = os.path.join(base_config.output_dir, "images")
    raw_prediction_dir = os.path.join(base_config.output_dir, "predictions")
    failed_tasks_dir = os.path.join(base_config.output_dir, "failed_tasks")

    os.makedirs(image_prediction_dir, exist_ok=True)
    os.makedirs(raw_prediction_dir, exist_ok=True)
    os.makedirs(failed_tasks_dir, exist_ok=True)

    tasks_path = os.path.join(
        base_config.dataset_dir, f"arc-agi_{base_config.dataset_category}_challenges.json"
    )

    with open(tasks_path, "rb") as f:
        tasks_challenges: dict = json.load(f)

    tasks_solutions: dict = {}
    if not base_config.kaggle_mode:
        solutions_path = os.path.join(
            base_config.dataset_dir, f"arc-agi_{base_config.dataset_category}_solutions.json"
        )
        with open(solutions_path, "rb") as f:
            tasks_solutions = json.load(f)

    subset_tasks = sorted(list(tasks_challenges.items()))[start_index_tasks:end_index_tasks]

    solved_tasks, total_tasks = 0, 0
    failed_tasks = {}
    for i, (task_name, demo_tasks) in enumerate(subset_tasks):
        output_dir = os.path.join(base_config.output_dir, f"{task_name}")
        try:
            save_original_model = os.path.join(output_dir, "original")
            save_merged_model = os.path.join(output_dir, "merged")

            if os.path.exists(f"{raw_prediction_dir}/submission_{task_name}.json"):
                logger.info(f"The task {task_name} is already attempted")
                continue

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(save_original_model, exist_ok=True)
            os.makedirs(save_merged_model, exist_ok=True)

            config = copy.deepcopy(base_config)
            config.output_dir = save_original_model

            wrapper_cls = MAP_WRAPPER[config.wrapper]
            wrapper = wrapper_cls(  # type: ignore
                model_id=config.model_id,
                gpu_index=gpu_index,
                quantization=config.quantization,
                online_finetuning=True,
            )
            peft_config = get_peft_config(
                target_modules=(
                    wrapper._target_modules
                    if config.lora_target_modules is None
                    else config.lora_target_modules
                ),
                dropout=config.lora_dropout,
                alpha=config.lora_alpha,
                r=config.lora_r,
            )

            sft_config = get_sft_config(config=config)

            print(f">>> Start training {task_name=}")

            logger.info("Initializing model")

            write_json(data=config.dict(), filename=f"{config.output_dir}/finetuning_config.json")

            callbacks = []  # type: ignore # [EarlyStoppingCallback(early_stopping_patience=10)]

            train_dataset = get_train_dataset(
                config=config,
                model_type=wrapper.model_type,
                grid_formatter=wrapper.grid_formatter,
                task_name=task_name,
                demo_tasks=demo_tasks,
            )

            logger.info("Length of train dataset: %d", len(train_dataset))

            eval_dataset = None
            if not config.kaggle_mode:
                eval_dataset = get_eval_dataset(
                    config=config,
                    model_type=wrapper.model_type,
                    grid_formatter=wrapper.grid_formatter,
                    task_name=task_name,
                    demo_tasks=demo_tasks,
                    test_solutions=tasks_solutions[task_name],
                )

            try:
                trainer = SFTTrainer(
                    model=wrapper.model,
                    args=sft_config,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=wrapper.collate_fn_train,
                    peft_config=peft_config,
                    callbacks=callbacks,
                )
            except ValueError as e:
                if "gradient checkpointing" in e.args[0]:
                    sft_config.gradient_checkpointing = False
                    trainer = SFTTrainer(
                        model=wrapper.model,
                        args=sft_config,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=wrapper.collate_fn_train,
                        peft_config=peft_config,
                        callbacks=callbacks,
                    )
                else:
                    raise ValueError(e)
            logger.info("Training model")
            trainer.train()
            logger.info("Saving model")
            trainer.model.to("cpu")
            trainer.save_model(config.output_dir)
            logger.info(f"Saving {wrapper.grid_formatter=}")
            wrapper.grid_formatter.save(config.output_dir)

            finetuned_config = OnlineFinetuningConfig.parse_file(
                f"{config.output_dir}/finetuning_config.json"
            )
            merge_model(
                finetuned_config, adaptor_path=config.output_dir, merge_path=save_merged_model  # type: ignore
            )

            subprocess.run(["rm", "-rf", config.output_dir])

            logger.info(" -- Evaluating Model --")

            model_config = copy.deepcopy(BASE_CONFIG)
            model_config["wrapper"] = wrapper_cls
            model_config["wrapper_kwargs"]["model_id"] = save_merged_model  # type: ignore
            model_config["evaluation_config"]["batch_size"] = config.eval_batch_size  # type: ignore
            model_config["evaluation_config"]["n_attempts"] = config.eval_n_attempts  # type: ignore
            model_config["evaluation_config"]["n_transforms"] = config.eval_n_transforms  # type: ignore
            model_config["evaluation_config"]["rigid_transforms_all"] = config.eval_rigid_transforms_all  # type: ignore
            model_config["evaluation_config"]["generation_config"][  # type: ignore
                "num_return_sequences"
            ] = config.eval_num_return_sequences
            model_config["evaluation_config"]["generation_config"][  # type: ignore
                "num_beams"
            ] = config.eval_num_beams

            logger.info(f"Config: {model_config}")

            if config.kaggle_mode:
                run_inference(  # type: ignore
                    logger,
                    task_name,
                    demo_tasks,
                    model_config,
                    f"{raw_prediction_dir}/submission_{task_name}.json",
                )
            else:
                solved, total = save_eval_results(  # type: ignore
                    logger,
                    task_name,
                    demo_tasks,
                    tasks_solutions[task_name],
                    model_config,
                    f"{raw_prediction_dir}/submission_{task_name}.json",
                    f"{image_prediction_dir}/{task_name}",
                )

                solved_tasks += solved
                total_tasks += total

                print("\033[92m" + f"SOLVED TASKS: {solved_tasks}/{total_tasks}" + "\033[0m")

        except Exception as e:
            print("\033[91m" + f"Error in task {task_name}" + "\033[0m")
            print("\033[91m" + str(e) + "\033[0m")
            # remove the dir of the task
            failed_tasks[task_name] = str(e)

        torch.cuda.empty_cache()
        # remove the original and merged models
        subprocess.run(["rm", "-rf", output_dir])

    with open(f"{failed_tasks_dir}/failed_tasks_gpu_{gpu_index}.json", "w") as f:
        json.dump(failed_tasks, f)


def is_main_process() -> bool:
    return "RANK" not in os.environ or int(os.environ["RANK"]) == 0


if __name__ == "__main__":
    # Causes warnings otherwise
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "HF_TOKEN" in os.environ:
        HF_TOKEN = os.environ["HF_TOKEN"]
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
    args = parse_arguments_main()
    config = OnlineFinetuningConfig(
        kaggle_mode=args["kaggle_mode"],
        model_id=args["model_id"],
        wrapper=args["wrapper"],
        dataset_dir=args["dataset_dir"],
        dataset_category=args["dataset_category"],
        output_dir=args["output_dir"],
        quantization=args["quantization"],
        transform_background_color=args["transform_background_color"],
        learning_rate=args["learning_rate"],
        per_device_batch_size=args["batch_size"],
        eval_batch_size=args["eval_batch_size"],
        eval_n_attempts=args["eval_n_attempts"],
        eval_n_transforms=args["eval_n_transforms"],
        eval_rigid_transforms_all=args["eval_rigid_transforms_all"],
        eval_num_return_sequences=args["eval_num_return_sequences"],
        eval_num_beams=args["eval_num_beams"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        num_train_epochs=args["num_train_epochs"],
        neftune_noise_alpha=args["neftune_noise_alpha"],
        padding_side=None,  # args["padding_side"],
        prompt_type=args["prompt_type"],
        lora_target_modules=args["lora_target_modules"],
        logging_steps=args["logging_steps"],
        eval_steps=args["eval_steps"],
        low_memory=args["low_memory"],
        lora_dropout=args["lora_dropout"],
        lora_alpha=args["lora_alpha"],
        lora_r=args["lora_r"],
        early_stopping_patience=args["early_stopping_patience"],
        save_total_limit=args["save_total_limit"],
        untie_word_embeddings=args["untie_word_embeddings"],
    )
    start_index_tasks = args["start_index_tasks"]
    end_index_tasks = args["end_index_tasks"]
    gpu_index = args["gpu_index"]
    main(
        logger=args["logger"],
        base_config=config,
        start_index_tasks=start_index_tasks,
        end_index_tasks=end_index_tasks,
        gpu_index=gpu_index,
    )
