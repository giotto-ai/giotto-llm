import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from ..type_aliases import ColorList, ColorSet, Grid, Ratio, Size, JSONTask

FOLDER_PATH = "data/all_800/"


def load_task(task_id: str) -> JSONTask:
    file_path = os.path.join(FOLDER_PATH, task_id + ".json")
    with open(file_path, "r") as file:
        task_data = json.load(file)
    return task_data


# Generic
def all_equal_but_one(xs: List[Any]) -> bool:
    if len(xs) == 0:
        return False
    value_counts = Counter(xs)
    if len(value_counts) == 1:
        return True
    if len(value_counts) > 2:
        return False
    (mc, mcv), (smc, smcv) = value_counts.most_common()
    if smcv == 1:
        return True
    else:
        return False


def all_zeros_but_one(xs: List[Any]) -> bool:
    if len(xs) == 0:
        return False
    value_counts = Counter(xs)
    if len(value_counts) == 1:
        return value_counts[0] > 0
    if len(value_counts) > 2:
        return False

    (mc, mcv), (smc, smcv) = value_counts.most_common()
    if mc == 0 and smcv == 1:
        return True
    else:
        return False


def dict_diff(d1: Dict[Any, int], d2: Dict[Any, int]) -> Dict[Any, int]:
    return {k: d1[k] - d2[k] for k in set(d1).union(d2)}


def dict_ratio(d1: Dict[Any, int], d2: Dict[Any, int]) -> Dict[Any, Tuple[int, int]]:
    return {k: reduced_ratio(d1[k], d2[k]) for k in set(d1).union(d2)}


def reduced_ratio(x: int, y: int) -> Tuple[int, int]:
    gcd = math.gcd(x, y)
    return (x // gcd, y // gcd)


# Color related functions
def get_all_colors_list(grid: Grid) -> ColorList:
    return sum(grid, [])


def get_all_colors_set(grid: Grid) -> ColorSet:
    return set(get_all_colors_list(grid))


def get_all_colors_counter(grid: Grid) -> Dict[int, int]:
    return Counter(get_all_colors_list(grid))


def get_color_diff(input_grid: Grid, output_grid: Grid) -> Dict[int, int]:
    input_color_count = get_all_colors_counter(input_grid)
    output_color_count = get_all_colors_counter(output_grid)
    return dict_diff(input_color_count, output_color_count)


def get_color_ratio(input_grid: Grid, output_grid: Grid) -> Dict[int, Tuple[int, int]]:
    input_color_count = get_all_colors_counter(input_grid)
    output_color_count = get_all_colors_counter(output_grid)
    return dict_ratio(input_color_count, output_color_count)


def almost_same_color_count(input_grid: Grid, output_grid: Grid) -> bool:
    color_count_diff = get_color_diff(input_grid, output_grid)
    # if len(color_count_diff) < 2:
    #    return False
    all_values = [v for _, v in color_count_diff.items()]
    return all_zeros_but_one(all_values)


def almost_same_color_ratio(input_grid: Grid, output_grid: Grid) -> bool:
    color_count_ratio = get_color_ratio(input_grid, output_grid)
    # Do not infer anything with less than 3
    # if len(color_count_ratio) < 2:
    #    return False
    all_values = [v for _, v in color_count_ratio.items()]
    return all_equal_but_one(all_values)


def remove_color_from_grid(grid: Grid, color_to_remove: int) -> Grid:
    grid_ = np.array(grid)
    grid_[grid_ == color_to_remove] = 11
    return grid_.tolist()


# def get_background_color(grid: Grid) -> int:
#     grid_ = [GridObj(np.array(grid, np.uint8), dsl.Pos(0, 0))]
#     return dsl.t_extract_bkg_color(grid_)[0].color


def get_non_background_mask(grid: Grid, color: int) -> Grid:
    return np.array(remove_color_from_grid(grid, color)) != 11


# Grid size related functions
def grid_size(grid: Grid) -> Size:
    return len(grid), len(grid[0])


def grids_ratio(input_grid: Grid, output_grid: Grid) -> Ratio:
    input_x, input_y = grid_size(input_grid)
    output_x, output_y = grid_size(output_grid)
    return (reduced_ratio(input_x, output_x), reduced_ratio(input_y, output_y))


def same_size(input_grid: Grid, output_grid: Grid) -> bool:
    return grid_size(input_grid) == grid_size(output_grid)


# Symmetry related functions
def vertical_flip(grid: Grid) -> Grid:
    grid_ = np.array(grid)
    return grid_[:, ::-1].tolist()


def has_vertical_symmetry(grid: Grid) -> bool:
    return bool(np.all(grid == vertical_flip(grid)))


def horizontal_flip(grid: Grid) -> Grid:
    grid_ = np.array(grid)
    return grid_[::-1, :].tolist()


def has_horizontal_symmetry(grid: Grid) -> bool:
    return bool(np.all(grid == horizontal_flip(grid)))


def transpose_flip(grid: Grid) -> Grid:
    grid_ = np.array(grid)
    return grid_.T.tolist()


def has_transpose_symmetry(grid: Grid) -> bool:
    x, y = grid_size(grid)
    if x != y:
        return False
    grid_ = np.array(grid)
    return bool(np.all(grid_ == grid_.T))


def antitranspose_flip(grid: Grid) -> Grid:
    grid_ = np.array(grid)
    return grid_[::-1].T[::-1].tolist()


def has_antitranspose_symmetry(grid: Grid) -> bool:
    x, y = grid_size(grid)
    if x != y:
        return False
    return bool(np.all(grid == antitranspose_flip(grid)))


def rot_90(grid: Grid) -> Grid:
    grid_ = np.array(grid)
    return grid_[::-1].T.tolist()


def has_90_rot_symmetry(grid: Grid) -> bool:
    x, y = grid_size(grid)
    if x != y:
        return False
    return bool(np.all(grid == rot_90(grid)))


def rot_180(grid: Grid) -> Grid:
    grid_ = np.array(grid)
    return grid_[::-1, ::-1].tolist()


def has_180_rot_symmetry(grid: Grid) -> bool:
    return bool(np.all(grid == rot_180(grid)))
