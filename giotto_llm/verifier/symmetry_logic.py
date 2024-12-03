from collections import defaultdict
from typing import Dict, List

from ..type_aliases import Example, Grid, LogicRule
from . import helper_functions

symmetry_functions = {
    "vertical_symmetry": helper_functions.has_vertical_symmetry,
    "horizontal_symmetry": helper_functions.has_horizontal_symmetry,
    "transpose_symmetry": helper_functions.has_transpose_symmetry,
    "antitranspose_symmetry": helper_functions.has_antitranspose_symmetry,
    "90_rot_symmetry": helper_functions.has_90_rot_symmetry,
    "180_rot_symmetry": helper_functions.has_180_rot_symmetry,
}


def compute_symmetries(grid: Grid) -> Dict[str, bool]:
    return {k: f(grid) for k, f in symmetry_functions.items()}


def example_logic(example: Example) -> Dict[str, LogicRule]:
    input_grid, output_grid = example["input"], example["output"]
    input_symmetries = compute_symmetries(input_grid)
    output_symmetries = compute_symmetries(output_grid)
    res = {
        # k: #not input_symmetries[k] and output_symmetries[k] for k in symmetry_functions
        k: output_symmetries[k]
        for k in symmetry_functions
    }
    # We add one more feature: if there is at least one symmetry in the input
    # and we have the same symmetry in the output, we want to remember this
    # we can then infer the symmetry from the output at verification time
    res["same_symmetry"] = any(input_symmetries.values()) and all(
        output_symmetries[k] == has_symmetry
        for k, has_symmetry in input_symmetries.items()
        if has_symmetry
    )
    return res


def build(examples: List[Example]) -> Dict[str, LogicRule]:
    res = defaultdict(list)
    for example in examples:
        for k, v in example_logic(example).items():
            res[k].append(v)
    # We need to have the same symmetry information across all examples
    new_rules = {k: all(vs) for k, vs in res.items()}
    return new_rules
