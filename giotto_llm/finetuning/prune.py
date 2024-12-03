import torch_pruning as tp
from transformers import PreTrainedModel

from ..consts import ROOT_PATH
from ..data import Dataset
from ..prompts.consts import TYPES_OF_PROMPTS
from ..prompts.grid_formatter import GridFormatter
from ..reader import ReaderMany
from ..transforms import Transforms
from ..utils import split_tasks_by_test
from .args import parse_arguments_prune
from .utils import MAP_WRAPPER


def get_dataset(model_type: str, grid_formatter: GridFormatter) -> Dataset:
    eval_tasks = ReaderMany(
        dataset_dir=f"{ROOT_PATH}/kaggle/input",
        dataset_type="evaluation",
        read_test_output=True,
    ).read_tasks()

    tasks = split_tasks_by_test(eval_tasks)

    dataset = Dataset(
        tasks=tasks,
        transforms=Transforms(),
        messages_fn=TYPES_OF_PROMPTS["prompt_solve_short"](grid_formatter=grid_formatter),
        model_type=model_type,  # type: ignore[arg-type]
    )
    return dataset


def prune_model(config: dict[str, str | float]) -> None:
    print("Loading model")
    assert isinstance(config["wrapper"], str)
    assert isinstance(config["model_id"], str)
    wrapper_cls = MAP_WRAPPER[config["wrapper"]]
    wrapper = wrapper_cls(
        model_id=config["model_id"], config={"quantization_config": None, "device_map": "auto"}
    )
    model = wrapper.model
    model.eval()

    # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Count the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    print("loading data")
    dataset = get_dataset(model_type=wrapper.model_type, grid_formatter=wrapper.grid_formatter)

    batch = wrapper.collate_fn_train([dataset[0]])  
    example_inputs = [batch["input_ids"], batch["attention_mask"]]

    ## 2. Initialize a pruner with the model and the importance criterion
    # ignored_layers = []
    # for name, m in model.named_modules():
    #    if isinstance(m, torch.nn.Linear) and "lm_head" in name:
    #        ignored_layers.append(m) # DO NOT prune the final classifier!
    # assert len(ignored_layers) == 1

    print("Setting up pruning")
    num_heads = {}
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            num_heads[m.q_proj] = model.config.num_attention_heads
            num_heads[m.k_proj] = model.config.num_key_value_heads
            num_heads[m.v_proj] = model.config.num_key_value_heads
    _is_gqa = model.config.num_attention_heads != model.config.num_key_value_heads
    importance = tp.importance.GroupNormImportance(p=2, group_reduction="mean")
    assert isinstance(config["prune_ratio"], float)
    assert isinstance(config["prune_steps"], int)
    assert isinstance(config["global_pruning"], bool)
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=config["global_pruning"],
        isomorphic=config["global_pruning"],
        pruning_ratio=config["prune_ratio"],
        ignored_layers=[model.lm_head],
        iterative_steps=config["prune_steps"],
        num_heads=num_heads,
        prune_num_heads=not config["global_pruning"],  # causes issues with global pruning
        prune_head_dims=False,
        head_pruning_ratio=config["prune_ratio"],
    )

    for i in range(config["prune_steps"]):
        print(f"Pruning step {i}")
        pruner.step()

    print("Updating model attributes")
    # Update model attributes
    model.config.hidden_size = model.lm_head.in_features
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            m.hidden_size = m.q_proj.out_features
            m.num_heads = m.hidden_size // m.head_dim
            model.config.num_attention_heads = m.num_heads
            # m.head_dim = m.q_proj.out_features // m.num_heads
            if not _is_gqa:
                m.num_key_value_heads = m.num_heads
            m.num_key_value_groups = m.num_heads // m.num_key_value_heads
        elif name.endswith("mlp"):
            model.config.intermediate_size = m.gate_proj.out_features
    if not _is_gqa:
        model.config.num_key_value_heads = model.config.num_attention_heads

    # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Count the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    print("Saving model")
    assert isinstance(config["prune_path"], str)
    wrapper.save_pretrained(config["prune_path"])


if __name__ == "__main__":
    config = parse_arguments_prune()
    prune_model(config)
