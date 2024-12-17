# giotto-llm

## Overview

This repo is a wrapper of many open source packages that are needed to fine-tune LLMs and run them in inference mode effectively.

**Important:** See the `Makefile` for useful commands.

## Documentation

The online documentation con be found at https://giotto-ai.github.io/giotto-llm/index.html

## Setup

### Environmental variables
The following environmental variables needs to be set (update credential path below):

```shell
GOOGLE_APPLICATION_CREDENTIALS=$HOME/.gloud/giotto-research-admin.json
export MLFLOW_TRACKING_URI=http://cluster-manager:5051
export PATH=$HOME/.local/bin:$PATH
```

### Gitlab ssh-key
Generate key with `ssh-keygen`, and add to gitlab.

### Install and download packages, download data and model

```shell
make
```

## Finetune models on multi-gpu
Example of QwenVL-based model:
```shell
torchrun --nproc-per-node=gpu -m llm_prompts.finetuning -d re_arc_400x5 --model_id Qwen/Qwen2-VL-2B-Instruct --wrapper QwenVL -o qwenvl --batch_size 1 --gradient_accumulation_steps 16 --quantization 8bit-4 --neftune_noise_alpha 10.0 --num_train_epochs 15 --learning_rate 2e-4
```
Example of Molmo-based model:
```shell
torchrun --nproc-per-node=gpu -m llm_prompts.finetuning -d re_arc_400x5 --model_id allenai/MolmoE-1B-0924 --wrapper Molmo -o molmo --batch_size 1 --gradient_accumulation_steps 16 --quantization 8bit-4 --neftune_noise_alpha 10.0 --num_train_epochs 15 --learning_rate 2e-4
```
Example of Llama-based models:
```shell
torchrun --nproc-per-node=gpu -m llm_prompts.finetuning -d re_arc_400x5 --model_id meta-llama/Llama-3.2-1B-Instruct --wrapper CausalLM -o llama --batch_size 1 --gradient_accumulation_steps 16 --quantization 8bit-4 --neftune_noise_alpha 10.0 --num_train_epochs 15 --learning_rate 2e-4
```
Example of Qwen-based models:
```shell
torchrun --nproc-per-node=gpu -m llm_prompts.finetuning -d re_arc_400x5 --model_id Qwen/Qwen2.5-0.5B --wrapper CausalLM -o qwen --batch_size 1 --gradient_accumulation_steps 16 --quantization 8bit-4 --neftune_noise_alpha 10.0 --num_train_epochs 15 --learning_rate 2e-4
```

See full list of currently tested models in `./models/`.

### Validate fine-tuned models

The validation script is single gpu for now, and requires a config entry in `./llm_prompts/validation/__main__.py`.
```shell
# Only single gpu support for now
CUDA_VISIBLE_DEVICES=0 python -m llm_prompts.validation --dataset_type evaluation --finetuned_model_id <MODEL-ID> --max_num_tasks 400
```

where `<MODEL-ID>` is a fine-tuned model defined in the config.

## Info
See

- [ARC technical guide](https://arcprize.org/guide): specs

```text
--- CPU only ---
4 CPU Cores
30 Gigabytes of RAM
```

or

```text
--- P100 GPU ---
1 Nvidia Tesla P100 GPI
4 CPU cores
29 Gigabytes of RAM
```

or

```text
--- T4 2x GPU ---
2 Nvidia Tesla T4 GPUs
4 CPU cores
29 Gigabytes of RAM
```

- [Awesome ARC](https://github.com/neoneye/arc-notes/tree/main/awesome): lots of information
    - [LLMs as a System of Multiple Expert Agents to solve the ARC Challenge (Detailed Walkthrough)](https://www.youtube.com/watch?v=sTvonsD5His)
    - [LLMs as a system to solve the Abstraction and Reasoning Corpus (ARC) Challenge!](https://www.youtube.com/watch?v=plVRxP8hQHY)
- [Lots of data](https://huggingface.co/neoneye): millions of synthetically generated tasks
- [Unsloth](https://github.com/unslothai/unsloth): fine-tune LLMs. Also with quantization.
- [Fine-tuning example](https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html): how to fine-tune and quantize a LLM model from HugginFace
