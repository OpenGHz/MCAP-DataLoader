from pydantic import BaseModel
from typing import Tuple, Dict, Union
from mcap_data_loader.callers.basis import CallerBasis, ReturnT


Item = Dict[str, ReturnT]


class DictTupleConfig(BaseModel):
    """Configuration for DictTuple pipeline."""

    depth: int = -1
    """Depth of tuple nesting to flatten.
    < 0 means auto-detect by whether the item is a tuple or not,
    which is useful for varying depth of tuples or the depth is unknown.
    A positive integer means flattening up to that depth, which requires
    the depth to be the same for all items and is faster. 0 means the items
    are already a flattened dictionary thus no further action will be taken.
    """
    separator: str = "/"
    """Separator used when concatenating prefixes."""
    separate_key: bool = True
    """Whether to separate the prefix and the dict key with a separator."""


class DictTuple(CallerBasis[Item]):
    """Convert a tuple of dictionaries into a single dictionary by flattening."""

    def __init__(self, config: DictTupleConfig):
        self._func = (
            self._process_auto
            if config.depth < 0
            else self._process_depth
            if config.depth > 0
            else lambda tp, prefix, depth: tp
        )
        self._sep = config.separator
        self._last_sep = config.separator if config.separate_key else ""
        self._depth = config.depth

    def __call__(self, data: Tuple[Item]):
        self._tuple_dict: Item = {}
        self._func(data, "", self._depth)
        return self._tuple_dict

    def _process_auto(self, tp: Tuple[Item], prefix: str, depth: int = 0):
        for i, value in enumerate(tp):
            if isinstance(value, tuple):
                self._process_auto(value, f"{prefix}{i}/")
            else:
                for k, v in value.items():
                    self._tuple_dict[f"{prefix}{i}/{k}"] = v

    def _process_depth(self, tp: Union[Tuple[Item], Item], prefix: str, depth: int):
        if depth > 1:
            for i, value in enumerate(tp):
                self._process_depth(value, f"{prefix}{i}{self._sep}", depth - 1)
        else:
            for i, value in enumerate(tp):
                for k, v in value.items():
                    self._tuple_dict[f"{prefix}{i}{self._last_sep}{k}"] = v


if __name__ == "__main__":
    import time

    print("---- Auto depth ----")

    for tuple_dict in [
        ({"a": 1}, {"b": 2}),
        ({"a": 1}, ({"b": 2}, {"c": 3})),
        ({"a": 1}, ({"b": 2}, ({"c": 3}, {"d": 4}))),
    ]:
        dict_tuple = DictTuple(DictTupleConfig())
        start = time.perf_counter()
        result = dict_tuple(tuple_dict)
        print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
        print(result)

    print("---- With depth ----")

    for index, tuple_dict in enumerate(
        [
            {"a": 1, "b": 2},
            ({"/a": 1}, {"/b": 2}),
            (({"/a": 1}, {"/b": 2}), ({"/c": 3}, {"/d": 4}, {"/e": 5})),
        ]
    ):
        dict_tuple = DictTuple(
            DictTupleConfig(depth=index, separator=".", separate_key=False)
        )
        start = time.perf_counter()
        result = dict_tuple(tuple_dict)
        print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
        print(result)
