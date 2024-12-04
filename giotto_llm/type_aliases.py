from typing import Any, Dict, List, Literal, Set, Tuple, TypeAlias, Union

Grid: TypeAlias = list[list[int]]
InputOutputPair: TypeAlias = dict[str, Grid]
JSONTask: TypeAlias = dict[str, list[InputOutputPair]]
Attempts: TypeAlias = dict[int, list[Grid]]

Messages: TypeAlias = list[dict[str, str]]
OAIMessage: TypeAlias = list[dict[str, Any]]

Response: TypeAlias = Tuple[bool, List[str]]
Example: TypeAlias = Dict[str, Grid]
Size: TypeAlias = Tuple[int, int]
Ratio: TypeAlias = Tuple[Tuple[int, int], Tuple[int, int]]
ColorList: TypeAlias = List[int]
ColorSet: TypeAlias = Set[int]
LogicRule: TypeAlias = Any  # Union[bool, Size, Ratio, ColorList, ColorSet]
ReducedRatio = Tuple[int, int]
RatioType = Tuple[ReducedRatio, ReducedRatio]
SizeType = Tuple[int, int]
