from collections import defaultdict
from typing import Dict, List, Tuple

from .helper_functions import (
    almost_same_color_count,
    almost_same_color_ratio,
    get_all_colors_list,
    get_all_colors_set,
    grid_size,
)
from ..type_aliases import Example, LogicRule

COLORS = [
    "Black",
    "Blue",
    "Red",
    "Green",
    "Yellow",
    "Grey",
    "Fuchsia",
    "Orange",
    "Cyan",
    "Brown",
]
DIGIT_TO_COLOR = {i: col for i, col in enumerate(COLORS)}


def example_logic(example: Example) -> Dict[str, LogicRule]:
    input_colors = get_all_colors_set(example["input"])
    output_colors = get_all_colors_set(example["output"])
    color_map, inverse_color_map = build_color_map(example)
    return {
        "fixed_colors": list(sorted(output_colors)),
        "new_colors": list(sorted(output_colors.difference(input_colors))),
        "dropped_colors": list(sorted(input_colors.difference(output_colors))),
        "all_colors": list(sorted(output_colors.union(input_colors))),
        "same_color_count": almost_same_color_count(
            example["input"], example["output"]
        ),
        "same_color_ratio": almost_same_color_ratio(
            example["input"], example["output"]
        ),
        "color_map": color_map,
        "inverse_color_map": inverse_color_map,
    }


def build_color_map(example: Example) -> Tuple[Dict[int, int], Dict[int, int]]:
    if grid_size(example["input"]) != grid_size(example["output"]):
        return dict(), dict()
    input_colors = get_all_colors_list(example["input"])
    output_colors = get_all_colors_list(example["output"])
    pre_color_map = defaultdict(set)
    pre_inverse_color_map = defaultdict(set)
    for ic, oc in zip(input_colors, output_colors):
        pre_color_map[ic].add(oc)
        pre_inverse_color_map[oc].add(ic)
    color_map = {
        k: head for k, (head, *tail) in pre_color_map.items() if len(tail) == 0
    }
    inverse_color_map = {
        k: head for k, (head, *tail) in pre_inverse_color_map.items() if len(tail) == 0
    }
    return color_map, inverse_color_map


def build(examples: List[Example]) -> Dict[str, LogicRule]:
    res = defaultdict(list)
    for example in examples:
        for k, v in example_logic(example).items():
            res[k].append(v)
    new_rules = {
        "new_colors": [],
        "dropped_colors": [],
        "fixed_colors": [],
        "same_colors": False,
        "never_seen_color": False,
        "same_color_count": False,
        "same_color_ratio": False,
        "has_color_map": False,
        "has_inverse_color_map": False,
    }
    if all(new_colors == res["new_colors"][0] for new_colors in res["new_colors"]):
        # for every example, we always have the same color(s) added in the output
        new_rules["new_colors"] = list(res["new_colors"][0])
    if all(
        dropped_colors == res["dropped_colors"][0]
        for dropped_colors in res["dropped_colors"]
    ):
        # for every example, we always have the same color(s) added in the output
        new_rules["dropped_colors"] = list(res["dropped_colors"][0])
    if all(
        output_colors == res["fixed_colors"][0] for output_colors in res["fixed_colors"]
    ):
        # for every example, we always have the same color(s) in the output
        new_rules["fixed_colors"] = list(res["fixed_colors"][0])
    new_rules["same_colors"] = all(
        len(new_colors) == 0 for new_colors in res["new_colors"]
    )
    new_rules["same_color_count"] = all(res["same_color_count"])
    new_rules["same_color_ratio"] = all(res["same_color_ratio"])
    new_rules["has_color_map"] = all(len(cm) > 0 for cm in res["color_map"])
    new_rules["has_inverse_color_map"] = all(
        len(icm) > 0 for icm in res["inverse_color_map"]
    )
    return new_rules
