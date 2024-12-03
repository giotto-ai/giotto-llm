from typing import Dict, List

from ..type_aliases import Example, Grid, JSONTask, LogicRule, Response
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
        wrong = []
        test_logic = symmetry_logic.example_logic({"input": input_grid, "output": output_grid})
        # We first check if input and output have the same symmetries. All
        # symmetries of the input must be present in the output, but the output
        # might have more symmetries. Otherwise we check symmetries of the
        # output per se.
        if not self.logic["any_inclusion"]:
            if self.logic["same_symmetry"]:
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
                if self.logic[k] and not has_symmetry:
                    wrong.append(k)
        return wrong

    def verify_size(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        wrong = []
        test_logic = size_logic.example_logic({"input": input_grid, "output": output_grid})
        ratio = helper_functions.grids_ratio(input_grid, output_grid)
        output_size = helper_functions.grid_size(output_grid)
        # We first check if we have a fixed ration. This is a stronger condition
        # than the fixed output size. We only resort of that if there is no
        # known ratio
        if self.logic["ratio"] is not None:
            if ratio != self.logic["ratio"]:
                wrong.append("ratio")
        elif self.logic["fixed_size"] is not None:
            if output_size != self.logic["fixed_size"]:
                wrong.append("fixed_size")
        return wrong

    def verify_color(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        wrong = []
        test_logic = color_logic.example_logic({"input": input_grid, "output": output_grid})
        if (
            len(self.logic["fixed_colors"])
            and not test_logic["fixed_colors"] == self.logic["fixed_colors"]
        ):
            wrong.append("fixed_colors")
        if (
            len(self.logic["new_colors"])
            and not test_logic["new_colors"] == self.logic["new_colors"]
        ):
            wrong.append("new_colors")
        if (
            len(self.logic["dropped_colors"])
            and not test_logic["dropped_colors"] == self.logic["dropped_colors"]
        ):
            wrong.append("dropped_colors")
        if self.logic["same_colors"] and len(test_logic["new_colors"]) > 0:
            wrong.append("same_colors")
        if self.logic["same_color_count"] and not test_logic["same_color_count"]:
            wrong.append("same_color_count")
        if self.logic["same_color_ratio"] and not test_logic["same_color_ratio"]:
            wrong.append("same_color_ratio")
        return wrong

    def verify_inclusion(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        wrong = []
        test_logic = inclusion_logic.example_logic({"input": input_grid, "output": output_grid})
        for k, has_inclusion in test_logic.items():
            if self.logic[k] and not has_inclusion:
                wrong.append(k)
        return wrong

    # def verify_shape(self, input_grid: Grid, output_grid: Grid) -> List[str]:
    #     wrong = []
    #     test_logic = shape_logic.example_logic(
    #         {"input": input_grid, "output": output_grid},
    #         background_color=self.logic["background_color"],
    #     )
    #     if (
    #         self.logic["non_background_stays_the_same"]
    #         and not test_logic["non_background_stays_the_same"]
    #     ):
    #         wrong.append("non_background_stays_the_same")
    #     if (
    #         self.logic["background_stays_the_same"]
    #         and not test_logic["background_stays_the_same"]
    #     ):
    #         wrong.append("background_stays_the_same")
    #     if not self.logic["any_inclusion"]:
    #         if self.logic["all_shape_matches"] and test_logic["shape_matches"] < 0.9:
    #             wrong.append("all_shape_matches")
    #         elif self.logic["shape_matches"] and test_logic["shape_matches"] == 0:
    #             wrong.append("shape_matches")
    #         if (
    #             self.logic["all_shape_and_color_matches"]
    #             and test_logic["shape_and_color_matches"] < 0.9
    #         ):
    #             wrong.append("all_shape_and_color_matches")
    #         elif (
    #             self.logic["shape_and_color_matches"]
    #             and test_logic["shape_and_color_matches"] == 0
    #         ):
    #             wrong.append("shape_and_color_matches")
    #         if (
    #             self.logic["all_shape_and_position_matches"]
    #             and test_logic["shape_and_position_matches"] < 0.9
    #         ):
    #             wrong.append("all_shape_and_position_matches")
    #         elif (
    #             self.logic["shape_and_position_matches"]
    #             and test_logic["shape_and_position_matches"] == 0
    #         ):
    #             wrong.append("shape_and_position_matches")
    #     return wrong

    def verify(self, input_grid: Grid, output_grid: Grid) -> List[str]:
        if not self.logic:
            self.build_logic()
        wrong_checks = []
        wrong_checks += self.verify_symmetry(input_grid, output_grid)
        wrong_checks += self.verify_size(input_grid, output_grid)
        wrong_checks += self.verify_color(input_grid, output_grid)
        wrong_checks += self.verify_inclusion(input_grid, output_grid)
        # wrong_checks += self.verify_shape(input_grid, output_grid)
        return wrong_checks


def get_hard_constraints(task_data: JSONTask, idx_i: int = 0) -> Dict[str, list]:
    res = {
        "size": set(),
        "colors": set(),
        "pixels": dict(),
    }
    test_input = task_data["test"][idx_i]["input"]
    # Size
    # We provide multiple options for the possible size of the output grid, since
    # one single logic might be too resctive. With one single exception, we always
    # guess the correct size in at most two tentatives.
    size = size_logic.build(task_data["train"])
    if (ratio := size["ratio"]) is not None:
        ratio_x, ratio_y = ratio
        input_x, input_y = helper_functions.grid_size(test_input)
        output_x = input_x * ratio_x[1] // ratio_x[0]
        output_y = input_y * ratio_y[1] // ratio_y[0]
        res["size"].add((output_x, output_y))
    if size["fixed_size"] is not None:
        res["size"].add(size["fixed_size"])
    res["size"] = list(res["size"])
    # We try to be as strict as possible with possible colors using pre-computed
    # logic. In case we have more information, we will take all possible sources
    # for colors. In some cases, this results in too many colors, but we never
    # miss a must-have color.
    colors = color_logic.build(task_data["train"])
    input_colors = helper_functions.get_all_colors_counter(test_input)
    all_output_colors = set()
    for example in task_data["train"]:
        all_output_colors.update(helper_functions.get_all_colors_set(example["output"]))
    if colors["same_colors"]:
        res["colors"].update(input_colors)
    if len(colors["new_colors"]) > 0:
        res["colors"].update(list(input_colors) + colors["new_colors"])
    if len(colors["dropped_colors"]) > 0:
        res["colors"].update([c for c in input_colors if c not in colors["dropped_colors"]])
    if len(colors["fixed_colors"]) > 0:
        res["colors"].update(colors["fixed_colors"])
    res["colors"].update(all_output_colors)
    res["colors"] = list(res["colors"])
    # If we know that the output is only an edit, e.g. only modying the
    # background part of the input, we know that everything else stays the same
    # shape = shape_logic.build(task_data["train"])
    # if shape["background_stays_the_same"] or shape["non_background_stays_the_same"]:
    #     bg_color = shape["background_color"]
    #     if bg_color is None:
    #         bg_color = helper_functions.get_background_color(test_input)
    #     output_pixel_maps = dict()
    #     for i, row in enumerate(test_input):
    #         for j, c in enumerate(row):
    #             if shape["background_stays_the_same"] and c == bg_color:
    #                 output_pixel_maps[(i, j)] = c
    #             if shape["non_background_stays_the_same"] and c != bg_color:
    #                 output_pixel_maps[(i, j)] = c
    #     res["pixels"] = output_pixel_maps
    return {k: v for k, v in res.items()}
