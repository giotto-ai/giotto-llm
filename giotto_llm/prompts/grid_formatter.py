import dataclasses
import json
import logging
import os

from transformers import AutoTokenizer

from giotto_llm.consts import (
    C0,
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
    C9,
    DEFAULT_END_EXAMPLE_TOKEN,
    DEFAULT_END_INPUT_TOKEN,
    DEFAULT_END_OUTPUT_TOKEN,
    DEFAULT_END_ROW_TOKEN,
    DEFAULT_START_EXAMPLE_TOKEN,
    DEFAULT_START_INPUT_TOKEN,
    DEFAULT_START_OUTPUT_TOKEN,
    DEFAULT_START_ROW_TOKEN,
    GRID_FORMATTER_CONFIG_FILENAME,
    LENGTH_STRING_COLOR_TOKEN,
    MAX_GRID_SIZE,
)
from giotto_llm.type_aliases import Grid


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class GridFormatter:
    sE_token: str = dataclasses.field(default=DEFAULT_START_EXAMPLE_TOKEN)
    eE_token: str = dataclasses.field(default=DEFAULT_END_EXAMPLE_TOKEN)
    sI_token: str = dataclasses.field(default=DEFAULT_START_INPUT_TOKEN)
    eI_token: str = dataclasses.field(default=DEFAULT_END_INPUT_TOKEN)
    sO_token: str = dataclasses.field(default=DEFAULT_START_OUTPUT_TOKEN)
    eO_token: str = dataclasses.field(default=DEFAULT_END_OUTPUT_TOKEN)
    sR_token: str = dataclasses.field(default=DEFAULT_START_ROW_TOKEN)
    eR_token: str = dataclasses.field(default=DEFAULT_END_ROW_TOKEN)
    c0: str = dataclasses.field(default=C0)
    c1: str = dataclasses.field(default=C1)
    c2: str = dataclasses.field(default=C2)
    c3: str = dataclasses.field(default=C3)
    c4: str = dataclasses.field(default=C4)
    c5: str = dataclasses.field(default=C5)
    c6: str = dataclasses.field(default=C6)
    c7: str = dataclasses.field(default=C7)
    c8: str = dataclasses.field(default=C8)
    c9: str = dataclasses.field(default=C9)
    color_separator_token: str = dataclasses.field(default="")

    def get_special_tokens_not_in(self, tokenizer: AutoTokenizer) -> list[str]:
        """Find which tokens need to be added to the tokenizer, when using this GridFromatter."""
        additional_special_tokens = []

        grid_formatting_tokens = [
            self.sE_token,
            self.eE_token,
            self.sI_token,
            self.eI_token,
            self.sO_token,
            self.eO_token,
            self.sR_token,
            self.eR_token,
            self.c0,
            self.c1,
            self.c2,
            self.c3,
            self.c4,
            self.c5,
            self.c6,
            self.c7,
            self.c8,
            self.c9,
            self.color_separator_token,
        ]

        vocab_dict = tokenizer.get_vocab()
        for token in grid_formatting_tokens:
            if token != "" and token not in vocab_dict:
                additional_special_tokens.append(token)


        return additional_special_tokens

    def encode_grid(self, grid: Grid, input_or_output: str) -> str:
        """Encode a Grid into a string to be used inside LLM prompts."""
        assert input_or_output in ["input", "output"]

        match input_or_output:
            case "input":
                s_t = self.sI_token
                e_t = self.eI_token
            case "output":
                s_t = self.sO_token
                e_t = self.eO_token

        map_colors = _get_map_color(
            color_tokens=[
                self.c0,
                self.c1,
                self.c2,
                self.c3,
                self.c4,
                self.c5,
                self.c6,
                self.c7,
                self.c8,
                self.c9,
            ]
        )

        num_rows = len(grid)
        formatted_grid_tokens = [s_t]
        for idx_row in range(num_rows):
            formatted_grid_tokens.append(self.sR_token)
            row = grid[idx_row]
            for color in row[:-1]:
                formatted_grid_tokens.append(map_colors[color])
                formatted_grid_tokens.append(self.color_separator_token)
            formatted_grid_tokens.append(map_colors[row[-1]])
            formatted_grid_tokens.append(self.eR_token)

        formatted_grid_tokens.append(e_t)

        return "".join(formatted_grid_tokens)

    def encode_pairs(self, pairs: list[dict[str, Grid]]) -> str:
        assert len(pairs) > 0
        formatted_pairs: str = ""

        for pair in pairs:
            formatted_example = self.sE_token

            encode_input_grid = self.encode_grid(pair["input"], input_or_output="input")
            formatted_example += encode_input_grid

            encode_output_grid = self.encode_grid(pair["output"], input_or_output="output")
            formatted_example += encode_output_grid

            formatted_example += self.eE_token

            formatted_pairs += formatted_example

        return formatted_pairs

    def decode_grid(
        self, str_containing_grid: str, input_or_output: str, logger: logging.Logger | None
    ) -> Grid | None:
        """Decode a strigng Grid into a list[list[int]]."""
        assert input_or_output in ["input", "output"]

        match input_or_output:
            case "input":
                s_t = self.sI_token
                e_t = self.eI_token
            case "output":
                s_t = self.sO_token
                e_t = self.eO_token

        reverse_map_colors = _get_reverse__map_color(
            color_tokens=[
                self.c0,
                self.c1,
                self.c2,
                self.c3,
                self.c4,
                self.c5,
                self.c6,
                self.c7,
                self.c8,
                self.c9,
            ]
        )

        try:
            start_index = str_containing_grid.find(s_t)
            end_index = (
                start_index + len(s_t) + str_containing_grid[start_index + len(s_t) :].find(e_t)
            )

            str_grid = str_containing_grid[start_index + len(s_t) : end_index]

            decoded_grid = []
            for row_with_start_token in str_grid.split(self.eR_token)[:-1]:
                row_tokens = row_with_start_token.split(self.sR_token)[1]
                row = []

                for idx in range(0, len(row_tokens), LENGTH_STRING_COLOR_TOKEN):
                    color_token = row_tokens[idx : idx + LENGTH_STRING_COLOR_TOKEN]
                    row.append(reverse_map_colors[color_token])

                decoded_grid.append(row)

            if len(decoded_grid) == 0:
                raise ValueError("Received empty grid")
            len_first_row = len(decoded_grid[0])
            for row_integers in decoded_grid:
                if len(row_integers) != len_first_row:
                    raise ValueError("Not same number of row elements")

            if len_first_row == 0:
                raise ValueError("Length of first row is zero")

            if len_first_row > MAX_GRID_SIZE:
                raise ValueError(f"Length of first row greater than {MAX_GRID_SIZE}")

            if len(decoded_grid) > MAX_GRID_SIZE:
                raise ValueError(f"Grid has more than {MAX_GRID_SIZE} rows")

        except Exception as e:
            if logger:
                logger.exception(f"{str(e)}")

            return None

        return decoded_grid

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, GRID_FORMATTER_CONFIG_FILENAME), "w") as f:
            json.dump(self.to_dict(), f)


def _get_map_color(color_tokens: list[str]) -> dict:
    return {idx: color_token for idx, color_token in enumerate(color_tokens)}


def _get_reverse__map_color(color_tokens: list[str]) -> dict:
    return {color_token: idx for idx, color_token in enumerate(color_tokens)}
