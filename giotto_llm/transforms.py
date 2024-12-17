from __future__ import annotations

import copy
from collections import defaultdict
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .type_aliases import Grid, JSONTask

N_TRAINING_OCCURANCE = np.asarray([0, 0, 101, 455, 166, 49, 19, 7, 2, 0, 1]) / 800

RIGID_TRANSFORMS: list[Callable[[list[list[int]]], list[list[int]]]] = [
    lambda x: x,
    lambda x: np.rot90(x, k=1).tolist(),
    lambda x: np.rot90(x, k=2).tolist(),
    lambda x: np.rot90(x, k=3).tolist(),
    lambda x: np.flipud(x).tolist(),
    lambda x: np.flipud(np.rot90(x, k=1)).tolist(),
    lambda x: np.flipud(np.rot90(x, k=2)).tolist(),
    lambda x: np.flipud(np.rot90(x, k=3)).tolist(),
]

# Lookup of which rigid transform is the inverse of a given index
REVERSE_RIGID_TRANSFORM_INDICES = np.asarray([0, 3, 2, 1, 4, 5, 6, 7])


class Transforms(BaseModel, frozen=True):
    """Transforms to apply to the task

    Args:
        test: randomly swap the test example with one of the training examples.
            This is done before any reordering.
        order: if 'reorder', reorder the training examples. If 'resample',
            sub- or super-sample the training data, such that the number of
            training examples is randomized.
        color: if 'all' randomize all colors. If 'foreground' randomize
            all but 0 (background)
        limit_colors: if `True`, use the smallest number of different colors possible.
        rigid: Do a random rigid transform (rotation + mirror)
        max_tokens: If given, subsample the task to keep the number of task tokens below
            this number (not counting test output)
    """

    test: bool = False
    order: Literal[None, "reorder", "resample"] = None
    color: Literal[None, "all", "foreground"] = None
    limit_colors: bool = False
    rigid: bool = False
    max_tokens: int | None = None


def _estimate_n_tokens_per_pair(task: JSONTask) -> list[int]:
    n_tokens = []
    for pair in task["train"]:
        n_rows_input = len(pair["input"])
        n_columns_input = len(pair["input"][0])
        n_rows_output = len(pair["output"])
        n_columns_output = len(pair["output"][0])
        n_tokens.append(
            6 + n_rows_input * (2 + n_columns_input) + n_rows_output * (2 + n_columns_output)
        )
    n_rows_input = len(task["test"][0]["input"])
    n_columns_input = len(task["test"][0]["input"][0])
    n_tokens.append(2 + n_rows_input * (2 + n_columns_input))
    return n_tokens


def transform_task(
    task: JSONTask, transforms: Transforms
) -> tuple[JSONTask, _BackTransformTestOutput]:
    """Do task transforms.

    Args:
        task: the un-transformed task
        transforms: the transforms to use

    Returns:
        transformed task
    """
    if len(task["test"]) != 1:
        msg = "Tasks should be pre-processed to only include 1 test pair each"
        raise ValueError(msg)
    n_training_examples = len(task["train"])
    back_transform: dict[str, bool | int | NDArray[np.int_]] = {}
    transformed_task = copy.deepcopy(task)
    back_transform["test"] = transforms.test
    if transforms.test is True:
        if np.random.random() > 1 / (n_training_examples + 1):
            idx = np.random.randint(0, n_training_examples)
            transformed_task["train"][idx], transformed_task["test"][0] = (
                transformed_task["test"][0],
                transformed_task["train"][idx],
            )

    match transforms.order:
        case "reorder":
            np.random.shuffle(transformed_task["train"])  # type: ignore[arg-type]
        case "resample":
            n_training = np.random.choice(range(len(N_TRAINING_OCCURANCE)), p=N_TRAINING_OCCURANCE)
            transformed_task["train"] = list(
                np.random.choice(
                    transformed_task["train"],  # type: ignore[arg-type]
                    size=n_training,
                    replace=(len(transformed_task["train"]) < n_training),
                )
            )

    if transforms.max_tokens is not None:
        n_tokens = _estimate_n_tokens_per_pair(transformed_task)
        n_train = len(n_tokens) - 1
        keep_n_train = 1
        for i in range(2, n_train + 1):
            if sum(n_tokens[:i]) + n_tokens[-1] <= transforms.max_tokens:
                keep_n_train = i
        transformed_task["train"] = transformed_task["train"][:keep_n_train]

    if transforms.color in ("all", "foreground"):
        color_transform_1 = np.arange(int(transforms.color == "foreground"), 10)
        np.random.shuffle(color_transform_1)
        if transforms.color == "foreground":
            color_transform_1 = np.concatenate([[0], color_transform_1])
        _transform_color_task(transformed_task, color_transform_1)
    else:
        color_transform_1 = np.arange(10)

    # This only tracks colors of test output for reconstruction
    color_transform_2 = np.arange(10)
    if transforms.limit_colors is True:
        color_summary = _get_color_summary(transformed_task)
        shared_colors = list(color_summary.shared_colors)
        n_shared_colors = len(shared_colors)
        for train_test in transformed_task:
            for i, pair in enumerate(transformed_task[train_test]):
                unique_pair_colors = list(color_summary.unique_pair_colors[train_test][i])
                n_unique_pair_colors = len(unique_pair_colors)
                # Need to make sure there's no duplicate colors
                remaining_colors = list(set(range(10)) - set(shared_colors + unique_pair_colors))
                color_transform = np.zeros(10, dtype=int)
                color_transform[shared_colors] = np.arange(n_shared_colors)
                color_transform[unique_pair_colors] = np.arange(
                    n_shared_colors, n_shared_colors + n_unique_pair_colors
                )
                color_transform[remaining_colors] = np.arange(
                    n_shared_colors + n_unique_pair_colors,
                    n_shared_colors + n_unique_pair_colors + len(remaining_colors),
                )
                for input_output in pair.keys():
                    colors = pair[input_output]
                    colors[:] = color_transform[colors].tolist()
                    if train_test == "test" and input_output == "input":
                        color_transform_2[:] = color_transform

    # Backtransform
    color_transform = np.zeros(10, dtype=int)
    color_transform[color_transform_2[color_transform_1]] = np.arange(10)
    back_transform["color_transform"] = color_transform

    if transforms.rigid is True:
        k = np.random.randint(0, 8)
        back_transform["rigid_index"] = k
        transformed_task = rigid_transform(transformed_task, k)

    return transformed_task, _BackTransformTestOutput(**back_transform)  # type: ignore[arg-type]


def rigid_transform(task: JSONTask, k: int) -> Grid:
    transformed_task = copy.deepcopy(task)
    for subset in transformed_task.values():
        for example in subset:
            for key, item in example.items():
                if len(item) == 0:
                    continue
                example[key] = RIGID_TRANSFORMS[k](item)
    return transformed_task


def backtransform_test_output(grid: Grid, backtransform: _BackTransformTestOutput) -> Grid:
    if backtransform.test is True:
        msg = "The test example was swapped with a training example during the transform."
        msg += "\n This should only happen during training, when backtransforms are not needed."
        raise RuntimeError(msg)

    backtransformed_grid: Grid = backtransform.color_transform[grid].tolist()

    k = REVERSE_RIGID_TRANSFORM_INDICES[backtransform.rigid_index]
    backtransformed_grid = RIGID_TRANSFORMS[k](backtransformed_grid)
    return backtransformed_grid


def _transform_color_task(task: JSONTask, color_transform: NDArray[np.int_]) -> None:
    """In-place color transform of given task.

    If `color_transform=[1, 2, 0]` then the following color transform will be applied:
        0 -> 1
        1 -> 2
        2 -> 0
    """
    for train_test in task:
        for pair in task[train_test]:
            for input_output in pair:
                colors = pair[input_output]
                colors[:] = color_transform[colors].tolist()


class _BackTransformTestOutput(BaseModel):
    """Data object with information needed to recover a transformed test output

    Args:
        test: if the test example was swapped with a training example.
        rigid_index: the index of the rigid transform
        color_transform: the applied color transform
    """

    test: bool
    color_transform: NDArray[np.int_]
    rigid_index: int = 0

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _ColorSummary(BaseModel):
    """Data object with summary of the colorization in the task.

    Args:
        shared_colors: the colors shared between examples
    """

    shared_colors: set[int]
    unique_pair_colors: dict[str, list[set[int]]]


def _get_color_summary(task: JSONTask) -> _ColorSummary:
    colors: dict[str, list[set[int]]] = defaultdict(list)
    for train_test in task:
        for pair in task[train_test]:
            joint_colors = set()
            for input_output in pair:
                if train_test == "test" and input_output == "output":
                    continue
                item_colors = set(np.asarray(pair[input_output]).ravel())
                colors[f"{train_test}_{input_output}"].append(item_colors)
                joint_colors |= item_colors
            colors[f"{train_test}_joint"].append(joint_colors)

    # Have to be robust here to prevent issues when output colors are not in input
    shared_colors_input = colors["train_input"][0]
    for example_colors in colors["train_input"] + colors["test_input"]:
        shared_colors_input &= example_colors
    shared_colors_output = colors["train_output"][0]
    for example_colors in colors["train_output"]:
        shared_colors_output &= example_colors
    shared_colors = shared_colors_input | shared_colors_output

    unique_pair_colors = {
        "train": [colors - shared_colors for colors in colors["train_joint"]],
        "test": [colors - shared_colors for colors in colors["test_input"]],
    }

    return _ColorSummary(shared_colors=shared_colors, unique_pair_colors=unique_pair_colors)
