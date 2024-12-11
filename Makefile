.PHONY: setup build install-packages test style bumpver bucket docker

all: setup install-packages download-data download-models

#------------------------------------------
# Setup and installing packages

setup:
	sudo apt update
	sudo apt upgrade --yes
	sudo apt install python3.10 python3.10-dev python3.10-venv python3-pip --yes
	sudo apt install tree htop nano --yes
	pip3 install -U pip
	pip3 install poetry==1.8.5

build:
	poetry build

install-packages:
	poetry install
	poetry run pip install --verbose --editable .

#------------------------------------------
# Utilities

test:
	pytest -v -rP giotto_llm/tests

style:
	python -m isort giotto_llm
	python -m black giotto_llm
	python -m mypy giotto_llm --config-file pyproject.toml

bumpver:
	python -m bumpver update -n --patch

#------------------------------------------
# Data

bucket:
	python -m giotto_llm.bucket list

get-synth-data:
	python -m giotto_llm.bucket.get_synth_data

download-data:
	python -m giotto_llm.bucket get synth_data/

generate-messages:
	python -m giotto_llm.generate_messages --output_file tmp.parquet --path_to_tokenizer giotto_llm/tests/llama_tokenizer --dataset_category re_arc_400x5 --max_seq_length 2048 --num_messages_to_generate 10 --prompts_with_weights prompt_solve_instr_v1:1,prompt_solve_instr_v2:1 --transforms_with_weights color_all_rigid_True:1

#------------------------------------------
# Validation

validation:
	python -m giotto_llm.validation --dataset_type full --finetuned_model_id models/llama/ID002_best_text_24_10_25_merged_pretrained_llama_1B_short_re_arc_400x200 --quantization no --batch_size 6 --n_attempts 2 --n_transforms 2 --num_beams 2 --start_index_tasks 0 --end_index_tasks 128 --random_seed 0 --gpu_index 3

#-------------------------------
# Validation: 30 October 2024
validation-multi-gpu-1:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID002_best_text_24_10_25_merged_pretrained_llama_1B_short_re_arc_400x200 --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0

validation-multi-gpu-2:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID004_merged_llama_1_drop_0_2_descr_v1_e100 --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0

validation-multi-gpu-3:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID005_merged_llama_1B_notr_untie_highdrop_check_1875 --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0


# Broken
validation-multi-gpu-llama-3b-1:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID006_merged_llama_3_2_3B_short_rearc --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0
# Broken
validation-multi-gpu-llama-3b-2:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID007_merged_llama_3_2_3B_short_rearc_lr_1e4_check_2048 --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0


validation-multi-gpu-automata-model-v1:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID008_merged_automata_model --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0


validation-multi-gpu-mekhron:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID009_merged_mekhron_v1/ --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2  --n_transforms 2 --num_beams 2 --random_seed 0


#-------------------------------
# Validation: 31 October 2024
validation-multi-gpu-wally_llama_1B_augmented_data_descr:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/merged_wally_llama_1B_augmented_data_descr --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0

validation-multi-gpu-ID010_merged_wally_online_finetuning_full:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID010_merged_wally_online_finetuning_full/ --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0

# Fixed
validation-multi-gpu-llama-3b-4:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/ID011_remerged_llama_3_2_3B_short_rearc_lr_1e4_check_2048/ --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0

validation-multi-gpu-wally_augmented_v2_check_2008:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/merged_wally_augmented_v2_check_2008/ --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0

validation-multi-gpu-merged_wally_arc_augmented_automata_basic_data_check_5394:
	python -m giotto_llm.parallelize.validation --dataset_type full --finetuned_model_id models/llama/merged_wally_arc_augmented_automata_basic_data_check_5394/ --wrapper_cls_type CausalLM --quantization no --batch_size 1 --n_attempts 2 --n_transforms 2 --num_beams 2 --random_seed 0



#------------------------------------------
# ML Ops
docker:
	@echo ">>> Setting up docker build prerequisites"
	./mlops/docker/setup-docker-build-prerequisites.sh
	@echo ">>> Building Docker image"
	cd mlops/docker && sudo docker build --tag=cluster-manager:5052/llm-finetune --progress=plain .


#------------------------------------------
# Finetuning models
download-models:
	python -m giotto_llm.bucket get models/llama/llama_3_2_1B_instruct/
	python -m giotto_llm.bucket get models/llama/llama_3_2_0.25B_instruct/
	python -m giotto_llm.bucket get models/llama/llama_3_2_0.5B_instruct/
	python -m giotto_llm.bucket get models/llama/ID001_best_text_24_10_25_adapters_pretrained_llama_1B_short_re_arc_400x200/
	python -m giotto_llm.bucket get models/llama/ID002_best_text_24_10_25_merged_pretrained_llama_1B_short_re_arc_400x200/
	python -m giotto_llm.bucket get models/llama/llama_3_1_8B_instruct/
	python -m giotto_llm.bucket get models/llama/llama_3_2_3B_instruct/
	python -m giotto_llm.bucket get models/llama/ID004_merged_llama_1_drop_0_2_descr_v1_e100/
	python -m giotto_llm.bucket get models/llama/ID005_merged_llama_1B_notr_untie_highdrop_check_1875/
	python -m giotto_llm.bucket get models/llama/ID006_merged_llama_3_2_3B_short_rearc/
	python -m giotto_llm.bucket get models/llama/ID007_merged_llama_3_2_3B_short_rearc_lr_1e4_check_2048/
	python -m giotto_llm.bucket get models/llama/ID009_merged_mekhron_v1/
	python -m giotto_llm.bucket get models/llama/ID010_merged_wally_online_finetuning_full/
	python -m giotto_llm.bucket get models/llama/ID011_remerged_llama_3_2_3B_short_rearc_lr_1e4_check_2048/

#------------------------------------------llama/llama_3_2_1B_instruct
# Pruning models
prune:
	python -m giotto_llm.finetuning.prune --model_id models/llama/ID002_best_text_24_10_25_merged_pretrained_llama_1B_short_re_arc_400x200 --wrapper CausalLM --prune_path models/llama/llama_3_2_1B_instruct_pruned_10_6 --prune_steps 10 --prune_ratio 0.6

save-base:
	python -m giotto_llm.finetuning.save_base_model --model_id meta-llama/Llama-3.2-1B-Instruct --wrapper CausalLM --output_model_dir models/llama/llama_3_2_1B_instruct


finetune-multi-gpu:
	torchrun --nproc-per-node=gpu -m giotto_llm.finetuning -d re_arc_400x5 --model_id meta-llama/Llama-3.2-1B-Instruct --wrapper CausalLM -o debug-finetuning --batch_size 1 --gradient_accumulation_steps 16 --quantization 8bit-4 --neftune_noise_alpha 10.0 --num_train_epochs 15 --learning_rate 2e-4

pretrain-multi-gpu:
	torchrun --nproc-per-node=gpu -m giotto_llm.finetuning.pretraining -d re_arc_400x5 --model_id meta-llama/Llama-3.2-1B-Instruct --wrapper CausalLM -o debug-pretraining --batch_size 1 --gradient_accumulation_steps 4 --quantization no --num_train_epochs 5 --learning_rate 1e-3 --evaluation_set_size 800 --prompt_type prompt_solve_short --early_stopping 2


#------------------------------------------
# Merging models

# 30 October 2024
merge-model-llama_1_drop_0_2_descr_v1_e100:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/llama_1_drop_0_2_descr_v1_e100 --merge_path ./models/llama/merged_llama_1_drop_0_2_descr_v1_e100

merge-model-llama_3_2_3B_short_rearc:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/llama_3_2_3B_short_rearc --merge_path ./models/llama/merged_llama_3_2_3B_short_rearc

merge-model-wally:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/llama_1B_notr_untie_highdrop --merge_path ./models/llama/merged_llama_1B_notr_untie_highdrop

merge-model-llama_3_2_3B_v2:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/llama_3_2_3B_short_rearc_lr_1e4_v2 --merge_path ./models/llama/merged_llama_3_2_3B_short_rearc_lr_1e4_v2

merge-model-automata_model:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/automata_model --merge_path ./models/llama/merged_automata_model


# 31 October 2024
merge-model-wally_llama_1B_augmented_data_descr:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/wally_llama_1B_augmented_data_descr --merge_path ./models/llama/merged_wally_llama_1B_augmented_data_descr

merge-model-wally_online_finetuning_full:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/wally_online_finetuning_full --merge_path ./models/llama/merged_wally_online_finetuning_full

merge-model-wally_augmented_v2_check_2008:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/wally_augmented_v2_check_2008 --merge_path ./models/llama/merged_wally_augmented_v2_check_2008

merge-model-llama_3_2_3B_short_rearc_5632:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/llama_3_2_3B_short_rearc_lr_1e4_v2 --merge_path ./models/llama/merged_llama_3_2_3B_short_rearc_5632

merge-model-wally_arc_augmented_automata_basic_data_check_5394:
	python -m giotto_llm.finetuning.merge --adaptor_path ./models/llama/wally_arc_augmented_automata_basic_data_check_5394 --merge_path ./models/llama/merged_wally_arc_augmented_automata_basic_data_check_5394
