import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Set, Tuple, Union, cast
from numpy.typing import NDArray

import numpy as np

from ..type_aliases import ColorList, ColorSet, Grid, JSONTask, Ratio, Size

FOLDER_PATH = "data/all_800/"


def load_task(task_id: str) -> JSONTask:
    file_path = os.path.join(FOLDER_PATH, task_id + ".json")
    with open(file_path, "r") as file:
        task_data = json.load(file)
    return cast(JSONTask, task_data)


# Generic Functions
def all_equal_but_one(xs: List[Any]) -> bool:
    if not xs:
        return False
    value_counts = Counter(xs)
    if len(value_counts) == 1:
        return True
    if len(value_counts) > 2:
        return False
    _, smcv = value_counts.most_common()[1]
    return smcv == 1


def all_zeros_but_one(xs: List[Any]) -> bool:
    if not xs:
        return False
    value_counts = Counter(xs)
    if len(value_counts) == 1:
        return 0 in value_counts
    if len(value_counts) > 2:
        return False
    (mc, _), (smc, smcv) = value_counts.most_common(2)
    return mc == 0 and smcv == 1


def dict_diff(d1: Dict[Any, int], d2: Dict[Any, int]) -> Dict[Any, int]:
    return {k: d1.get(k, 0) - d2.get(k, 0) for k in set(d1) | set(d2)}


def reduced_ratio(x: int, y: int) -> Tuple[int, int]:
    if y == 0:
        return (x, 0)
    gcd = math.gcd(x, y)
    return (x // gcd, y // gcd)


def dict_ratio(d1: Dict[Any, int], d2: Dict[Any, int]) -> Dict[Any, Tuple[int, int]]:
    return {k: reduced_ratio(d1.get(k, 0), d2.get(k, 0)) for k in set(d1) | set(d2)}


# Color-related Functions
def get_all_colors_list(grid: Grid) -> ColorList:
    return [color for row in grid for color in row]


def get_all_colors_set(grid: Grid) -> ColorSet:
    return set(get_all_colors_list(grid))


def get_all_colors_counter(grid: Grid) -> Dict[int, int]:
    return Counter(get_all_colors_list(grid))


def get_color_diff(input_grid: Grid, output_grid: Grid) -> Dict[int, int]:
    input_color_count = get_all_colors_counter(input_grid)
    output_color_count = get_all_colors_counter(output_grid)
    return dict_diff(input_color_count, output_color_count)


def get_color_ratio(
    input_grid: Grid, output_grid: Grid
) -> Dict[int, Tuple[int, int]]:
    input_color_count = get_all_colors_counter(input_grid)
    output_color_count = get_all_colors_counter(output_grid)
    return dict_ratio(input_color_count, output_color_count)


def almost_same_color_count(input_grid: Grid, output_grid: Grid) -> bool:
    color_count_diff = get_color_diff(input_grid, output_grid)
    all_values = list(color_count_diff.values())
    return all_zeros_but_one(all_values)


def almost_same_color_ratio(input_grid: Grid, output_grid: Grid) -> bool:
    color_count_ratio = get_color_ratio(input_grid, output_grid)
    all_values = list(color_count_ratio.values())
    return all_equal_but_one(all_values)


def remove_color_from_grid(grid: Grid, color_to_remove: int) -> Grid:
    grid_array = np.array(grid, dtype=int)
    grid_array[grid_array == color_to_remove] = 11
    return cast(Grid, grid_array.tolist())


def get_non_background_mask(grid: Grid, color: int) -> NDArray[np.bool_]:
    grid_array: NDArray[np.int_] = np.array(remove_color_from_grid(grid, color), dtype=int)
    mask: NDArray[np.bool_] = grid_array != 11
    return mask


# Grid Size-related Functions
def grid_size(grid: Grid) -> Size:
    return len(grid), len(grid[0])


def grids_ratio(input_grid: Grid, output_grid: Grid) -> Tuple[Ratio, Ratio]:
    input_x, input_y = grid_size(input_grid)
    output_x, output_y = grid_size(output_grid)
    return (
        reduced_ratio(input_x, output_x),
        reduced_ratio(input_y, output_y),
    )


def same_size(input_grid: Grid, output_grid: Grid) -> bool:
    return grid_size(input_grid) == grid_size(output_grid)


# Symmetry-related Functions
def vertical_flip(grid: Grid) -> Grid:
    grid_array = np.array(grid, dtype=int)
    flipped_array = grid_array[:, ::-1]
    return cast(Grid, flipped_array.tolist())


def has_vertical_symmetry(grid: Grid) -> bool:
    grid_array = np.array(grid, dtype=int)
    return bool(np.array_equal(grid_array, grid_array[:, ::-1]))


def horizontal_flip(grid: Grid) -> Grid:
    grid_array = np.array(grid, dtype=int)
    flipped_array = grid_array[::-1, :]
    return cast(Grid, flipped_array.tolist())


def has_horizontal_symmetry(grid: Grid) -> bool:
    grid_array = np.array(grid, dtype=int)
    return bool(np.array_equal(grid_array, grid_array[::-1, :]))


def transpose_flip(grid: Grid) -> Grid:
    grid_array = np.array(grid, dtype=int)
    transposed_array = grid_array.T
    return cast(Grid, transposed_array.tolist())


def has_transpose_symmetry(grid: Grid) -> bool:
    rows, cols = grid_size(grid)
    if rows != cols:
        return False
    grid_array = np.array(grid, dtype=int)
    return bool(np.array_equal(grid_array, grid_array.T))


def antitranspose_flip(grid: Grid) -> Grid:
    grid_array = np.array(grid, dtype=int)
    antitransposed_array = grid_array[::-1].T[::-1]
    return cast(Grid, antitransposed_array.tolist())


def has_antitranspose_symmetry(grid: Grid) -> bool:
    rows, cols = grid_size(grid)
    if rows != cols:
        return False
    grid_array = np.array(grid, dtype=int)
    return bool(np.array_equal(grid_array, antitranspose_flip(grid)))


def rot_90(grid: Grid) -> Grid:
    grid_array = np.array(grid, dtype=int)
    rotated_array = np.rot90(grid_array, k=3)
    return cast(Grid, rotated_array.tolist())


def has_90_rot_symmetry(grid: Grid) -> bool:
    rows, cols = grid_size(grid)
    if rows != cols:
        return False
    grid_array = np.array(grid, dtype=int)
    return bool(np.array_equal(grid_array, np.rot90(grid_array, k=3)))


def rot_180(grid: Grid) -> Grid:
    grid_array = np.array(grid, dtype=int)
    rotated_array = np.rot90(grid_array, k=2)
    return cast(Grid, rotated_array.tolist())


def has_180_rot_symmetry(grid: Grid) -> bool:
    grid_array = np.array(grid, dtype=int)
    return bool(np.array_equal(grid_array, np.rot90(grid_array, k=2)))
