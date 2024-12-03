from typing import Any, Dict, List, Set, Tuple

from ..type_aliases import Example, Grid, JSONTask, LogicRule
from . import color_logic, helper_functions, inclusion_logic, size_logic, symmetry_logic


class Verifier:
    def __init__(
        self,
        task_data: JSONTask,
    ):
        self.task_data: JSONTask = task_data
        self.train: List[Example] = task_data["train"]
        self.logic: Dict[str, LogicRule] = dict()
        self.build_logic()

    def build_logic(self) -> Dict[str, LogicRule]:
        # Base logic
        self.logic.update(symmetry_logic.build(self.train))
        self.logic.update(size_logic.build(self.train))
        self.logic.update(color_logic.build(self.train))
        self.logic.update(inclusion_logic.build(self.train))
        # self.logic.update(shape_logic.build(self.train))

        return self.logic

    def compact_logic(self) -> Dict[str, LogicRule]:
        return {k: v for k, v in self.logic.items() if v}

    # Verifiers
    def verify_symmetry(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        wrong: List[str] = []
        test_logic = symmetry_logic.example_logic({"input": input_grid, "output": output_grid})
        # We first check if input and output have the same symmetries. All
        # symmetries of the input must be present in the output, but the output
        # might have more symmetries. Otherwise we check symmetries of the
        # output per se.
        if not self.logic.get("any_inclusion", False):
            if self.logic.get("same_symmetry", False):
                test_output_symmetries = symmetry_logic.compute_symmetries(output_grid)
                test_input_symmetries = symmetry_logic.compute_symmetries(input_grid)
                if any(
                    test_input_symmetries[k] and not test_output_symmetries[k]
                    for k in test_input_symmetries
                ):
                    wrong.append("same_symmetry")
                    return wrong
            for k, has_symmetry in test_logic.items():
                if k == "same_symmetry":
                    continue
                if self.logic.get(k, False) and not has_symmetry:
                    wrong.append(k)
        return wrong

    def verify_size(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        wrong: List[str] = []
        test_logic = size_logic.example_logic({"input": input_grid, "output": output_grid})
        ratio = helper_functions.grids_ratio(input_grid, output_grid)
        output_size = helper_functions.grid_size(output_grid)
        # We first check if we have a fixed ratio. This is a stronger condition
        # than the fixed output size. We only resort to that if there is no
        # known ratio
        if self.logic.get("ratio") is not None:
            if ratio != self.logic["ratio"]:
                wrong.append("ratio")
        elif self.logic.get("fixed_size") is not None:
            if output_size != self.logic["fixed_size"]:
                wrong.append("fixed_size")
        return wrong

    def verify_color(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        wrong: List[str] = []
        test_logic = color_logic.example_logic({"input": input_grid, "output": output_grid})
        if (
            self.logic.get("fixed_colors")
            and test_logic["fixed_colors"] != self.logic["fixed_colors"]
        ):
            wrong.append("fixed_colors")
        if (
            self.logic.get("new_colors")
            and test_logic["new_colors"] != self.logic["new_colors"]
        ):
            wrong.append("new_colors")
        if (
            self.logic.get("dropped_colors")
            and test_logic["dropped_colors"] != self.logic["dropped_colors"]
        ):
            wrong.append("dropped_colors")
        if self.logic.get("same_colors") and len(test_logic["new_colors"]) > 0:
            wrong.append("same_colors")
        if self.logic.get("same_color_count") and not test_logic["same_color_count"]:
            wrong.append("same_color_count")
        if self.logic.get("same_color_ratio") and not test_logic["same_color_ratio"]:
            wrong.append("same_color_ratio")
        return wrong

    def verify_inclusion(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        wrong: List[str] = []
        test_logic = inclusion_logic.example_logic({"input": input_grid, "output": output_grid})
        for k, has_inclusion in test_logic.items():
            if self.logic.get(k) and not has_inclusion:
                wrong.append(k)
        return wrong

    def verify(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        if not self.logic:
            self.build_logic()
        wrong_checks: List[str] = []
        wrong_checks += self.verify_symmetry(input_grid, output_grid)
        wrong_checks += self.verify_size(input_grid, output_grid)
        wrong_checks += self.verify_color(input_grid, output_grid)
        wrong_checks += self.verify_inclusion(input_grid, output_grid)
        return wrong_checks


def get_hard_constraints(task_data: JSONTask, idx_i: int = 0) -> Dict[str, Any]:
    res_size: Set[Tuple[int, int]] = set()
    res_colors: Set[int] = set()
    res_pixels: Dict[Tuple[int, int], int] = {}
    test_input = task_data["test"][idx_i]["input"]
    # Size
    size = size_logic.build(task_data["train"])
    if (ratio := size.get("ratio")) is not None:
        ratio_x, ratio_y = ratio
        input_x, input_y = helper_functions.grid_size(test_input)
        output_x = input_x * ratio_x[1] // ratio_x[0]
        output_y = input_y * ratio_y[1] // ratio_y[0]
        res_size.add((output_x, output_y))
    if size.get("fixed_size") is not None:
        res_size.add(size["fixed_size"])
    # Convert sets to lists
    res_size_list = list(res_size)
    # Colors
    colors = color_logic.build(task_data["train"])
    input_colors = helper_functions.get_all_colors_counter(test_input)
    all_output_colors: Set[int] = set()
    for example in task_data["train"]:
        all_output_colors.update(helper_functions.get_all_colors_set(example["output"]))
    if colors.get("same_colors"):
        res_colors.update(input_colors.keys())
    if len(colors.get("new_colors", [])) > 0:
        res_colors.update(input_colors.keys())
        res_colors.update(colors["new_colors"])
    if len(colors.get("dropped_colors", [])) > 0:
        res_colors.update([c for c in input_colors if c not in colors["dropped_colors"]])
    if len(colors.get("fixed_colors", [])) > 0:
        res_colors.update(colors["fixed_colors"])
    res_colors.update(all_output_colors)
    # Convert sets to lists
    res_colors_list = list(res_colors)
    # Construct the result
    res: Dict[str, Any] = {
        "size": res_size_list,
        "colors": res_colors_list,
        "pixels": res_pixels,
    }
    return res
