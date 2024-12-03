import os
from typing import Callable

import pytest

from giotto_llm.causal_lm.models import CausalLMWrapper
from giotto_llm.consts import (
    C2,
    C3,
    C7,
    C8,
    DEFAULT_ATTEMPT,
    DEFAULT_END_OUTPUT_TOKEN,
    DEFAULT_END_ROW_TOKEN,
    DEFAULT_START_OUTPUT_TOKEN,
    DEFAULT_START_ROW_TOKEN,
    ROOT_PATH,
)
from giotto_llm.data import Dataset, task_to_image, task_to_oai_causal_lm, task_to_oai_vision
from giotto_llm.prompts.consts import TYPES_OF_PROMPTS
from giotto_llm.prompts.grid_formatter import GridFormatter
from giotto_llm.prompts.text_prompts import TextPromptBase
from giotto_llm.reader import ReaderMany
from giotto_llm.transforms import Transforms, transform_task
from giotto_llm.type_aliases import JSONTask, OAIMessage
from giotto_llm.utils import split_tasks_by_test

DEFAULT_TASKS = {
    "_": {
        "train": [{"input": DEFAULT_ATTEMPT, "output": DEFAULT_ATTEMPT}],
        "test": [{"input": DEFAULT_ATTEMPT, "output": DEFAULT_ATTEMPT}],
    }
}
DIR_PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "tasks", [{"dataset_type": "evaluation", "read_test_output": False}], indirect=True
)
def test_task_to_image(tasks: dict[str, JSONTask]) -> None:
    transforms = Transforms(limit_colors=True)
    reader = ReaderMany(
        dataset_dir=f"{ROOT_PATH}/kaggle/input",
        dataset_type="evaluation",
    )
    tasks = reader.read_tasks()
    tasks = split_tasks_by_test(tasks)
    for task_id, task in tasks.items():
        transformed_task = transform_task(task=task, transforms=transforms)[0]
        task_to_image(transformed_task)


@pytest.mark.parametrize(
    "tasks", [{"dataset_type": "evaluation", "read_test_output": True}], indirect=True
)
@pytest.mark.parametrize("fn", [task_to_oai_vision, task_to_oai_causal_lm])
def test_task_to_oai(
    tasks: dict[str, JSONTask], fn: Callable[[JSONTask, TextPromptBase], OAIMessage]
) -> None:
    task_id = "00576224"
    task = tasks[task_id]
    messages_fn = TYPES_OF_PROMPTS["prompt_solve_short"](grid_formatter=GridFormatter())
    message = fn(task, messages_fn)
    system, user, assistant = message

    SG, EG = DEFAULT_START_OUTPUT_TOKEN, DEFAULT_END_OUTPUT_TOKEN
    SR, ER = DEFAULT_START_ROW_TOKEN, DEFAULT_END_ROW_TOKEN
    expected_assistant_text = (
        f"{SG}{SR}{C3}{C2}{C3}{C2}{C3}{C2}{ER}"
        f"{SR}{C7}{C8}{C7}{C8}{C7}{C8}{ER}"
        f"{SR}{C2}{C3}{C2}{C3}{C2}{C3}{ER}"
        f"{SR}{C8}{C7}{C8}{C7}{C8}{C7}{ER}"
        f"{SR}{C3}{C2}{C3}{C2}{C3}{C2}{ER}"
        f"{SR}{C7}{C8}{C7}{C8}{C7}{C8}{ER}{EG}"
    )
    if fn is task_to_oai_vision:
        assert len(user["content"]) == 2
        assert assistant["content"][0]["text"] == expected_assistant_text
    else:
        assert assistant["content"] == expected_assistant_text


@pytest.mark.parametrize(
    "dataset",
    [
        Dataset(tasks=DEFAULT_TASKS, model_type="text-to-text"),
        Dataset(
            tasks=DEFAULT_TASKS,
            transforms=Transforms(limit_colors=True),
            model_type="image-text-to-text",
        ),
    ],
)
def test_dataset(dataset: Dataset) -> None:
    """Test that the different Dataset classes can return items successfully"""
    data = dataset[0]
    oai_message = data[0]
    keys = ["system", "user", "assistant"]
    for i, item in enumerate(oai_message):
        assert len(item) == 2
        assert item["role"] == keys[i]
        assert "content" in item


@pytest.mark.parametrize("mask", [True, False])
def test_label_mask(mask: bool, tasks: dict[str, JSONTask], wrapper: CausalLMWrapper) -> None:
    """Test that the label masking works as expected"""
    task_id = "fafffa47"
    tasks = {task_id: tasks[task_id]}
    dataset = Dataset(tasks=tasks, model_type="text-to-text")
    batch = wrapper.collate_fn_train([dataset[0]], mask_inputs=mask)
    if mask is True:
        assert batch["labels"][0, 0] == -100
    else:
        assert batch["labels"][0, 0] != -100
