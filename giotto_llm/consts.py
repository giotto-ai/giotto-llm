import pathlib

from giotto_llm.transforms import Transforms
from giotto_llm.type_aliases import Grid

ROOT_PATH = pathlib.Path().cwd()

NUMBER_OF_COLORS = 10
MAX_GRID_SIZE = 30

EMPTY_GRID: Grid = []
DEFAULT_ATTEMPT: Grid = [[0, 0], [0, 0]]

TYPES_OF_TRANSFORMS = {
    "color_foreground_limit_rigid": Transforms(
        test=False, order="reorder", color="foreground", limit_colors=True, rigid=True
    ),
    "color_foreground_rigid": Transforms(
        test=False, order="reorder", color="foreground", limit_colors=False, rigid=True
    ),
    "color_all": Transforms(
        test=False, order="reorder", color="all", limit_colors=False, rigid=False
    ),
    "color_all_rigid": Transforms(
        test=False, order="reorder", color="all", limit_colors=False, rigid=True
    ),
}

GRID_FORMATTER_CONFIG_FILENAME = "grid_formatter_config.json"
DATA_CONFIG_FILENAME = "data_config.json"
# Note: the following tokens were placed here to avoid a circular import

# Pair of grids delimiters
DEFAULT_START_EXAMPLE_TOKEN = "<|start_example|>"
DEFAULT_END_EXAMPLE_TOKEN = "<|end_example|>"

# Input grid delimiters
DEFAULT_START_INPUT_TOKEN = "<|start_input|>"
DEFAULT_END_INPUT_TOKEN = "<|end_input|>"

# Ouput grid delimiters
DEFAULT_START_OUTPUT_TOKEN = "<|start_output|>"
DEFAULT_END_OUTPUT_TOKEN = "<|end_output|>"

# Row delimiters
DEFAULT_START_ROW_TOKEN = "<|start_row|>"
DEFAULT_END_ROW_TOKEN = "<|end_row|>"

# Color tokens
C0 = "<|color_0|>"
C1 = "<|color_1|>"
C2 = "<|color_2|>"
C3 = "<|color_3|>"
C4 = "<|color_4|>"
C5 = "<|color_5|>"
C6 = "<|color_6|>"
C7 = "<|color_7|>"
C8 = "<|color_8|>"
C9 = "<|color_9|>"
COLOR_TOKENS = [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9]

# Note: do not delete the folliwing check
# Used in the parsing of a Grid
LENGTH_STRING_COLOR_TOKEN = len(C0)
for idx in range(len(COLOR_TOKENS)):
    assert LENGTH_STRING_COLOR_TOKEN == len(COLOR_TOKENS[idx])
