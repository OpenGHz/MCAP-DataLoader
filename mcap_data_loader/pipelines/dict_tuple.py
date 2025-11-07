from pydantic import BaseModel
from collections.abc import Generator
from mcap_data_loader.pipelines.basis import Pipeline, T
from typing import Tuple, Dict, Union


Item = Dict[str, T]


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


class DictTuple(Pipeline[Tuple[Item]]):
    """Convert a tuple of dictionaries into a single dictionary by flattening."""

    def __init__(self, config: DictTupleConfig) -> None:
        self.config = config
        self._func = (
            self._process_auto
            if config.depth < 0
            else self._process_depth
            if config.depth > 0
            else lambda tp, prefix, depth: tp
        )
        self._last_sep = config.separator if config.separate_key else ""

    def __iter__(self) -> Generator[Item]:
        for item in self._iterable:
            self._tuple_dict = {}
            yield self._func(item, "", self.config.depth)

    def _process_auto(self, tp: Tuple[Item], prefix: str, depth: int = 0) -> Item:
        for i, value in enumerate(tp):
            if isinstance(value, tuple):
                self._process_auto(value, f"{prefix}{i}/")
            else:
                for k, v in value.items():
                    self._tuple_dict[f"{prefix}{i}/{k}"] = v
        return self._tuple_dict

    def _process_depth(
        self, tp: Union[Tuple[Item], Item], prefix: str, depth: int
    ) -> Item:
        if depth > 1:
            for i, value in enumerate(tp):
                self._process_depth(
                    value, f"{prefix}{i}{self.config.separator}", depth - 1
                )
        else:
            for i, value in enumerate(tp):
                for k, v in value.items():
                    self._tuple_dict[f"{prefix}{i}{self._last_sep}{k}"] = v
        return self._tuple_dict


if __name__ == "__main__":
    import time

    print("---- Auto depth ----")

    for tuple_dict in [
        ({"a": 1}, {"b": 2}),
        ({"a": 1}, ({"b": 2}, {"c": 3})),
        ({"a": 1}, ({"b": 2}, ({"c": 3}, {"d": 4}))),
    ]:
        dict_tuple = DictTuple(DictTupleConfig())([tuple_dict])
        start = time.perf_counter()
        result = next(iter(dict_tuple))
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
        )([tuple_dict])
        start = time.perf_counter()
        result = next(iter(dict_tuple))
        print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
        print(result)
