from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..type_aliases import Example, Grid, LogicRule
from . import helper_functions

transformations = {
    "identity": lambda x: x,
    "vertical_flip": helper_functions.vertical_flip,
    "horizontal_flip": helper_functions.horizontal_flip,
    "transpose_flip": helper_functions.transpose_flip,
    "antitranspose_flip": helper_functions.antitranspose_flip,
    "90_rot": helper_functions.rot_90,
    "180_rot": helper_functions.rot_180,
}


def does_first_contain_second(first_grid: Grid, second_grid: Grid) -> bool:
    first_grid_ = np.array(first_grid)
    second_grid_ = np.array(second_grid)
    first_x, first_y = first_grid_.shape
    second_x, second_y = second_grid_.shape
    if second_x > first_x or second_y > first_y:
        return False
    for x in range(first_x - second_x + 1):
        for y in range(first_y - second_y + 1):
            subgrid = first_grid_[x : x + second_x, y : y + second_y]
            if (subgrid == second_grid_).all():
                return True
    return False


def does_first_contain_second_with_symmetries(first_grid: Grid, second_grid: Grid) -> bool:
    checks = []
    for k, transformation in transformations.items():
        second_grid_transformed = transformation(second_grid)
        checks.append(does_first_contain_second(first_grid, second_grid_transformed))
    return any(checks)


def example_logic(example: Example) -> Dict[str, LogicRule]:
    input_grid, output_grid = example["input"], example["output"]
    return {
        "input_contains_output": does_first_contain_second(input_grid, output_grid),
        "output_contains_input": does_first_contain_second(output_grid, input_grid),
        "input_contains_output_with_symmetries": does_first_contain_second_with_symmetries(
            input_grid, output_grid
        ),
        "output_contains_input_with_symmetries": does_first_contain_second_with_symmetries(
            output_grid, input_grid
        ),
    }


def build(examples: List[Example]) -> Dict[str, LogicRule]:
    res = defaultdict(list)
    for example in examples:
        for k, v in example_logic(example).items():
            res[k].append(v)
    new_rules = {k: all(vs) for k, vs in res.items()}
    new_rules["any_inclusion"] = any(new_rules.values())
    return new_rules
