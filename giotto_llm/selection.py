import hashlib
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from giotto_llm.verifier import inclusion_logic
from giotto_llm.verifier.verifier import get_hard_constraints

from giotto_llm.transforms import RIGID_TRANSFORMS
from giotto_llm.type_aliases import Grid, JSONTask


def select_top_2(
    attempts: list[Grid],
    log_probs: list[float],
    task: JSONTask,
    weight_method: str,
    threshold: float,
    constraints: dict[str, Any] = {},
):
    """Select the top 2 attempts"""

    # -------------------
    # DO NOT CHANGE
    threshold = 0.5
    # ------------------
    if len(attempts) < 2:
        return attempts
    attempts, log_probs = _filter_attempts_with_constraints(attempts, log_probs, task, constraints)
    if len(attempts) < 2:
        return attempts

    match weight_method:
        case "uniform":
            weights = np.ones(len(attempts)) / len(attempts)
        case "ll_sum":
            weights = torch.softmax(
                torch.tensor([ll.sum() if ll is not None else -torch.inf for ll in log_probs]),
                dim=0,
            ).numpy()
        case "entropy":
            weights = torch.softmax(
                torch.tensor(
                    [
                        (torch.exp(ll) * ll).mean() if ll is not None else -torch.inf
                        for ll in log_probs
                    ]
                ),
                dim=0,
            ).numpy()

    # First to full grid majority voting
    subset = _full_grid_majority_vote(attempts=attempts, weights=weights, threshold=threshold)
    # Then pixelwise
    if len(subset) < 2:
        _pixelwise_majority_vote(attempts, weights, subset)
    # Finally sorting by weight
    if len(subset) < 2:
        for attempt in [attempts[idx] for idx in np.argsort(weights)[::-1]]:
            if attempt not in subset and len(subset) < 2:
                subset.append(attempt)

    # Sanity checks (TODO remove)
    assert len(subset) <= 2
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            assert subset[i] != subset[j]

    return subset


def _full_grid_majority_vote(attempts, weights, threshold: float):
    mapping = _group_same([_numpy_to_hash(attempt) for attempt in attempts])
    score = {}
    for key, indices in mapping.items():
        score[key] = sum(weights[idx] for idx in indices)
    best_key_idx = np.argsort(list(score.values()))[::-1][:2]
    keys = list(score.keys())
    best_keys = [keys[idx] for idx in best_key_idx]
    subset = [attempts[mapping[key][0]] for key in best_keys if score[key] >= threshold]
    return subset


def _pixelwise_majority_vote(attempts, weights, subset):
    # first select most common shape
    # then vote per pixel
    mapping = _group_same([np.asarray(attempt).shape for attempt in attempts])
    score = {}
    for key, indices in mapping.items():
        score[key] = sum(weights[idx] for idx in indices)
    best_key_indices = np.argsort(list(score.values()))[::-1][:2]
    keys = list(score.keys())
    # NOTE: choose different shapes for attempts, but could do
    #       logic to only change a few pixels
    for key_indices in best_key_indices:
        shape = keys[key_indices]
        indices = mapping[keys[key_indices]]
        attempts_ = [attempts[idx] for idx in indices]
        pixel_attempt = np.zeros(shape, dtype=np.uint8)
        for idx_0 in range(shape[0]):
            for idx_1 in range(shape[1]):
                pixel_mapping = _group_same([attempt[idx_0][idx_1] for attempt in attempts_])
                pixel_score = {}
                for key, indices in pixel_mapping.items():
                    pixel_score[key] = sum(weights[idx] for idx in indices)
                    pixel_best_key_indices = np.argsort(list(pixel_score.values()))[-1]
                    pixel_keys = list(pixel_score.keys())
                    best_pixel = pixel_keys[pixel_best_key_indices]
                    pixel_attempt[idx_0, idx_1] = best_pixel
        pixel_attempt_ = pixel_attempt.tolist()
        if pixel_attempt_ not in subset:
            subset.append(pixel_attempt_)
        if len(subset) == 2:
            break


def _filter_attempts_with_constraints(attempts, log_probs, task, constraints):
    if len(constraints) == 0:
        constraints = get_hard_constraints(task, 0)
    constraints |= inclusion_logic.build(task["train"])
    filtered_attempts = []
    filtered_log_probs = []
    for attempt, log_prob in zip(attempts, log_probs, strict=True):
        output = np.asarray(attempt)
        colors = set(output.ravel().tolist())
        # print(constraints)
        # print(output.shape)
        # print(colors)
        if len(constraints["colors"]) > 0 and not colors.issubset(constraints["colors"]):
            continue

        if len(constraints["size"]) > 0 and output.shape not in [
            tuple(sublist) for sublist in constraints["size"]
        ]:
            continue
        input_ = np.asarray(task["test"][0]["input"])
        if constraints["input_contains_output"] is True:
            if _is_subset(subset_array=output, main_array=input_) is False:
                continue
        else:
            if _is_subset(subset_array=output, main_array=input_) is True:
                continue
            if constraints["input_contains_output_with_symmetries"] is True:
                for transform in RIGID_TRANSFORMS[1:]:
                    if (
                        _is_subset(subset_array=np.asarray(transform(output)), main_array=input_)
                        is True
                    ):
                        break
                else:
                    continue
            else:
                is_valid = True
                for transform in RIGID_TRANSFORMS[1:]:
                    if (
                        _is_subset(subset_array=np.asarray(transform(output)), main_array=input_)
                        is True
                    ):
                        is_valid = False
                        break
                if is_valid is False:
                    continue
        if constraints["output_contains_input"] is True:
            if _is_subset(subset_array=input_, main_array=output) is False:
                continue
        else:
            if _is_subset(subset_array=input_, main_array=output) is True:
                continue
            if constraints["output_contains_input_with_symmetries"] is True:
                for transform in RIGID_TRANSFORMS:
                    if (
                        _is_subset(subset_array=np.asarray(transform(input_)), main_array=output)
                        is True
                    ):
                        break
                else:
                    continue
            else:
                is_valid = True
                for transform in RIGID_TRANSFORMS:
                    if (
                        _is_subset(subset_array=np.asarray(transform(input_)), main_array=output)
                        is True
                    ):
                        is_valid = False
                        break
                if is_valid is False:
                    continue
        for indices, color in constraints.get("pixels", {}).items():
            if (
                len(attempt) < indices[0]
                and len(attempt[0]) < indices[1]
                and attempt[indices[0]][indices[1]] != color
            ):
                break
        else:
            filtered_attempts.append(attempt)
            filtered_log_probs.append(log_prob)

    # dead code
    # if len(filtered_attempts) == 0 and False:
    #     return attempts, log_probs
    return filtered_attempts, filtered_log_probs


def _is_subset(subset_array, main_array):
    # Get the shapes of the main array and the subset array
    subset_shape = subset_array.shape
    main_shape = main_array.shape

    # Ensure the subset array is not larger than the main array
    if subset_shape[0] > main_shape[0] or subset_shape[1] > main_shape[1]:
        return False

    # Slide over main_array to find a matching subarray
    for i in range(main_shape[0] - subset_shape[0] + 1):
        for j in range(main_shape[1] - subset_shape[1] + 1):
            # Extract a subarray of the same shape as subset_array
            if np.array_equal(
                main_array[i : i + subset_shape[0], j : j + subset_shape[1]], subset_array
            ):
                return True  # Match found

    return False  # No match found


def _numpy_to_hash(x: NDArray[np.int8] | list[list[int]]) -> str:
    """Convert numpy array to sha1 hash"""
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.int8)).hexdigest()  # type: ignore[arg-type]


def _group_same(items):
    mapping = {}
    for i, item in enumerate(items):
        if item not in mapping:
            mapping[item] = []
        mapping[item].append(i)
    return mapping
