from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..type_aliases import Example, JSONTask, LogicRule,RatioType,SizeType
from . import helper_functions


def input_output_have_always_same_size(task: JSONTask) -> bool:
    ratio = size_based_logic(task)["ratio"]
    # Ensure ratio is not None and compare with ((1, 1), (1, 1))
    return ratio == ((1, 1), (1, 1))

def example_logic(example: Example) -> Dict[str, LogicRule]:
    input_grid, output_grid = example["input"], example["output"]
    input_x, input_y = helper_functions.grid_size(input_grid)
    output_x, output_y = helper_functions.grid_size(output_grid)
    return {
        "ratio": helper_functions.grids_ratio(input_grid, output_grid),
        "fixed_size": (output_x, output_y),
    }

def size_based_logic(task: JSONTask) -> Dict[str, Optional[LogicRule]]:
    res: Dict[str, Optional[LogicRule]] = {
        "ratio": None,
        "fixed_size": None,
    }
    ratios: Set[RatioType] = set()
    output_sizes: Set[SizeType] = set()
    for example in task["train"]:
        input_grid, output_grid = example["input"], example["output"]
        ratio = helper_functions.grids_ratio(input_grid, output_grid)
        ratios.add(ratio)
        output_sizes.add(helper_functions.grid_size(output_grid))
    if ratios:
        ratios_xs = [rx for rx, _ in ratios]
        ratios_ys = [ry for _, ry in ratios]
        if len(set(ratios_xs)) == 1 and len(set(ratios_ys)) == 1:
            res["ratio"] = (ratios_xs[0], ratios_ys[0])  # This is of type RatioType
    if output_sizes:
        output_xs, output_ys = zip(*output_sizes)
        if len(set(output_xs)) == 1 and len(set(output_ys)) == 1:
            res["fixed_size"] = (output_xs[0], output_ys[0])  # This is of type SizeType
    return res

def build(examples: List[Example]) -> Dict[str, Optional[LogicRule]]:
    res: Dict[str, Set[LogicRule]] = defaultdict(set)
    for example in examples:
        for k, v in example_logic(example).items():
            res[k].add(v)
    new_rules: Dict[str, Optional[LogicRule]] = {
        "ratio": next(iter(res["ratio"])) if len(res["ratio"]) == 1 else None,
        "fixed_size": next(iter(res["fixed_size"])) if len(res["fixed_size"]) == 1 else None,
    }
    return new_rules
