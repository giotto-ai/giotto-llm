from collections import defaultdict
from typing import Dict, List

from ..type_aliases import Example, JSONTask, LogicRule
from . import helper_functions


def input_output_have_always_same_size(task: JSONTask) -> bool:
    return size_based_logic(task)["ratio"] == (1, 1)


def example_logic(example: Example) -> Dict[str, LogicRule]:
    input_grid, output_grid = example["input"], example["output"]
    input_x, input_y = helper_functions.grid_size(input_grid)
    output_x, output_y = helper_functions.grid_size(output_grid)
    return {
        "ratio": helper_functions.grids_ratio(input_grid, output_grid),
        "fixed_size": (output_x, output_y),
    }


def size_based_logic(task: JSONTask) -> Dict[str, LogicRule]:
    res = {
        "ratio": ((-1, -1), (-1, -1)),
        "fixed_size": (-1, -1),
    }
    ratios, output_sizes = set(), set()
    for example in task["train"]:
        input_grid, output_grid = example["input"], example["output"]
        ratios.add(helper_functions.grids_ratio(input_grid, output_grid))
        output_sizes.add(helper_functions.grid_size(output_grid))
    ratios_xs, ratios_ys = zip(*ratios)
    output_xs, output_ys = zip(*output_sizes)
    if len(ratios_xs) == 1 and len(ratios_ys) == 1:
        res["ratio"] = (list(ratios_xs)[0], list(ratios_ys)[0])
    elif len(output_xs) == 1 and len(output_ys) == 1:
        res["fixed_size"] = (next(iter(output_xs)), next(iter(output_ys)))
    return res


def build(examples: List[Example]) -> Dict[str, LogicRule]:
    res = defaultdict(set)
    for example in examples:
        for k, v in example_logic(example).items():
            res[k].add(v)
    new_rules = {
        "ratio": (next(iter(res["ratio"])) if len(res["ratio"]) == 1 else None),
        "fixed_size": (next(iter(res["fixed_size"])) if len(res["fixed_size"]) == 1 else None),
    }
    return new_rules
