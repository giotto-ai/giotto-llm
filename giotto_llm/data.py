"""
Data.
"""

from functools import partial
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from PIL import Image

from .consts import DEFAULT_ATTEMPT
from .prompts.consts import TYPES_OF_PROMPTS
from .prompts.grid_formatter import GridFormatter
from .prompts.text_prompts import PromptSolveShort, TextPromptBase
from .transforms import Transforms, _BackTransformTestOutput, transform_task
from .type_aliases import JSONTask, OAIMessage
from .utils import split_tasks_by_test
from .verifier import get_hard_constraints

# Keep "mixed colors" last so they are used less often
# They are selected to maximize the dot product to the
# pure colors
RGB_COLORS = np.asarray(
    [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (76, 239, 99),
        (99, 76, 239),
        (239, 99, 76),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)


class Dataset(torch.utils.data.Dataset):
    """Dataset class for tasks.

    Will split datasets to only have 1 test example automatically.
    Converts the task into an OAIMessage that can be collated by
    model-specific collate functions.

    Args:
        tasks: the tasks
        transforms: the random transforms to apply to the tasks
        messages_fn: the prompt class
    """

    def __init__(
        self,
        tasks: dict[str, JSONTask],
        transforms: Transforms = Transforms(),
        messages_fn: TextPromptBase = PromptSolveShort(grid_formatter=GridFormatter()),
        model_type: Literal["image-text-to-text", "text-to-text"] = "image-text-to-text",
        constraints_strategy: Literal["no", "token_subset", "valid"] = "valid",
    ):

        self.tasks = split_tasks_by_test(tasks)
        self.keys = list(self.tasks.keys())
        self.transforms = transforms
        self.size = len(self.tasks)
        self.messages_fn = messages_fn
        self.model_type = model_type
        if model_type == "image-text-to-text" and transforms.limit_colors is False:
            raise RuntimeError("Image-models need to use the limit_colors transform")
        self.task_to_oai = (
            partial(task_to_oai_vision)
            if model_type == "image-text-to-text"
            else task_to_oai_causal_lm
        )
        self.constraints_strategy = constraints_strategy

    def __len__(self) -> int:
        """The size of the dataset."""
        return self.size

    def __getitem__(
        self, idx: int
    ) -> tuple[OAIMessage, JSONTask, dict[str, Any], int, _BackTransformTestOutput]:
        """Get transformed task in OAIMessage format

        Returns:
            OAIMessage, transformed task, task index and the backtransform
        """
        task = self.tasks[self.keys[idx]]
        constraints = get_hard_constraints(task, 0) if self.constraints_strategy == "valid" else {}
        transformed_task, backtransform = transform_task(task=task, transforms=self.transforms)
        if self.add_task_id is True:
            transformed_task["task_id"] = self.keys[idx]  # type: ignore[assignment]

        oai_message = self.task_to_oai(task=transformed_task, messages_fn=self.messages_fn)
        return oai_message, transformed_task, constraints, idx, backtransform


class ConversationDataset(torch.utils.data.Dataset):
    """Dataset class for conversations.

    Args:
        parquet_filename: the parquet file with conversations
        max_token_length: the maximum token length to allow
        model_type: the model type (image-text-to-text or text-to-text)
    """

    def __init__(
        self,
        parquet_filename: str,
        max_token_length: int = 1_000_000,
        model_type: Literal["image-text-to-text", "text-to-text"] = "image-text-to-text",
    ):
        self.data = (
            pl.scan_parquet(parquet_filename)
            .filter(pl.col("length_tokenized_text") <= max_token_length)
            .select(["messages"] + ["task", "transforms"] * (model_type == "image-text-to-text"))
            .collect()
        )

        self.series_to_oai = (
            partial(series_to_oai_vision)
            if model_type == "image-text-to-text"
            else series_to_oai_causal_lm
        )
        self.size = len(self.data)
        # Dummy transform
        self.backtransform = _BackTransformTestOutput(test=False, color_transform=np.arange(10))
        # Dummy task
        self.transformed_task: JSONTask = {
            "train": [{"input": DEFAULT_ATTEMPT, "output": DEFAULT_ATTEMPT}],
            "test": [{"input": DEFAULT_ATTEMPT, "output": DEFAULT_ATTEMPT}],
        }

    def __len__(self) -> int:
        """The size of the dataset."""
        return self.size

    def __getitem__(
        self, idx: int
    ) -> tuple[OAIMessage, JSONTask, dict[str, Any], int, _BackTransformTestOutput]:
        """Get message.

        Returns:
            OAIMessage, task index and a dummy backtransform
        """
        series = self.data[idx]
        oai_message = self.series_to_oai(self.data[idx])
        return oai_message, self.transformed_task, {}, idx, self.backtransform


def task_to_image(task: JSONTask) -> Image.Image:
    """Convert a task to an image format.

    The image is RGB and uses (255, 255, 255) to indicate unused pixels.
    """

    n_train = len(task["train"])
    image = np.zeros((30 * (n_train + 1), 2 * 30, 3), dtype=np.uint8) + 255
    grid = np.asarray(task["test"][0]["input"], dtype=np.uint8)
    image[: grid.shape[0], : grid.shape[1]] = RGB_COLORS[grid]
    for i in range(1, n_train + 1):
        for j, key in enumerate(("input", "output")):
            grid = np.asarray(task["train"][i - 1][key], dtype=np.uint8)
            image[i * 30 : i * 30 + grid.shape[0], j * 30 : j * 30 + grid.shape[1]] = RGB_COLORS[
                grid
            ]
    return Image.fromarray(image)


def task_to_oai_vision(task: JSONTask, messages_fn: TextPromptBase) -> OAIMessage:
    """Convert task to OAIMessage"""
    messages = messages_fn(task=task, idx_i=0)
    assert len(messages) == 3

    return _create_oai_vision_message(
        task=task,
        system_message=messages[0]["content"],
        user_message=messages[1]["content"],
        assistant_message=messages[2]["content"],
    )


def task_to_oai_causal_lm(task: JSONTask, messages_fn: TextPromptBase) -> OAIMessage:
    """Convert task to OAIMessage"""
    messages = messages_fn(task=task, idx_i=0)
    assert len(messages) == 3

    return _create_oai_message(
        system_message=messages[0]["content"],
        user_message=messages[1]["content"],
        assistant_message=messages[2]["content"],
    )


def series_to_oai_vision(series: pl.DataFrame) -> OAIMessage:
    """Convert series to OAIMessage"""
    transforms = series["transform"].item()
    message = series["messages"].item()
    task = series["task"].item()
    return _create_oai_vision_message(
        task=task,
        system_message=message[0]["content"],
        user_message=message[1]["content"],
        assistant_message=message[2]["content"],
    )


def series_to_oai_causal_lm(series: pl.DataFrame) -> OAIMessage:
    """Convert task to OAIMessage"""
    message = series["messages"].item()
    return _create_oai_message(
        system_message=message[0]["content"],
        user_message=message[1]["content"],
        assistant_message=message[2]["content"],
    )


def _create_oai_vision_message(
    task: JSONTask,
    system_message: str,
    user_message: str,
    assistant_message: str,
) -> OAIMessage:
    image = task_to_image(task)

    message: OAIMessage = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message,
                },
                {
                    "type": "image",
                    "image": image,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_message}],
        },
    ]
    return message


def _create_oai_message(
    system_message: str, user_message: str, assistant_message: str
) -> OAIMessage:
    message = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": assistant_message,
        },
    ]
    return message
