from pydantic import BaseModel, NonNegativeInt
from collections.abc import Generator
from mcap_data_loader.pipelines.basis import Pipeline, T
from typing import Tuple, Dict


Item = Dict[str, T]


class DictTupleConfig(BaseModel):
    """Configuration for DictTuple pipeline."""

    depth: NonNegativeInt = 0
    """Depth of tuple nesting to flatten. 
    0 means auto-detect by whether the item is a tuple or not,
    which is useful for varying depth of tuples or the depth is unknown.
    A positive integer means flattening up to that depth, which requires
    the depth to be the same for all items and is faster.
    """


class DictTuple(Pipeline[Tuple[Item]]):
    """Convert a tuple of dictionaries into a single dictionary by flattening."""

    def __init__(self, config: DictTupleConfig) -> None:
        self.config = config
        self._func = self._process_auto if config.depth == 0 else self._process_depth

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

    def _process_depth(self, tp: Tuple[Item], prefix: str, depth: int) -> Item:
        if depth > 1:
            for i, value in enumerate(tp):
                self._process_depth(value, f"{prefix}{i}/", depth - 1)
        else:
            for i, value in enumerate(tp):
                for k, v in value.items():
                    self._tuple_dict[f"{prefix}{i}/{k}"] = v
        return self._tuple_dict


if __name__ == "__main__":
    import time

    print("---- Auto depth ----")

    for tuple_dict in [
        ({"a": 1}, {"b": 2}),
        ({"a": 1}, ({"b": 2}, {"c": 3})),
        ({"a": 1}, ({"b": 2}, ({"c": 3}, {"d": 4}))),
    ]:
        dict_tuple = DictTuple(DictTupleConfig(depth=0))([tuple_dict])
        start = time.perf_counter()
        result = next(iter(dict_tuple))
        print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
        print(result)

    print("---- With depth ----")

    for index, tuple_dict in enumerate(
        [
            ({"a": 1}, {"b": 2}),
            (({"a": 1}, {"b": 2}), ({"c": 3}, {"d": 4}, {"e": 5})),
        ]
    ):
        dict_tuple = DictTuple(DictTupleConfig(depth=index + 1))([tuple_dict])
        start = time.perf_counter()
        result = next(iter(dict_tuple))
        print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
        print(result)
