from pydantic import BaseModel, ConfigDict
from typing import Tuple, Dict, Union
from mcap_data_loader.callers.basis import CallerBasis, ReturnT


Item = Dict[str, ReturnT]


class DictTupleConfig(BaseModel, frozen=True):
    """Configuration for DictTuple caller."""

    model_config = ConfigDict(extra="forbid")

    depth: int = -1
    """Depth of tuple nesting to flatten.
    < 0 means auto-detect by whether the item is a tuple or not,
    which is useful for varying depth of tuples or the depth is unknown.
    A positive integer means flattening up to that depth, which requires
    the depth to be the same for all items and is faster. 0 means the items
    are already a flattened dictionary thus no further action will be taken.
    """
    separator: str = "."
    """Separator used when concatenating prefixes."""
    separate_key: bool = True
    """Whether to separate the prefix and the dict key with a separator."""
    zero_prefix: bool = False
    """Whether to add a prefix for the zeroth level. If False, `0` prefix will not be added."""


class DictTuple(CallerBasis[Item]):
    """Convert a tuple of dictionaries into a single dictionary by flattening."""

    def __init__(self, config: DictTupleConfig):
        self._func = (
            self._process_auto
            if config.depth < 0
            else self._process_depth
            if config.depth > 0
            else lambda tp, prefix, depth, rm_zero: tp
        )
        self._sep = config.separator
        self._last_sep = config.separator if config.separate_key else ""
        self._depth = config.depth
        self._rm_zero = not config.zero_prefix

    def __call__(self, data: Tuple[Item]):
        self._tuple_dict: Item = {}
        self._func(data, "", self._depth, self._rm_zero)
        return self._tuple_dict

    def _process_auto(
        self, tp: Tuple[Item], prefix: str, depth: int = 0, rm_zero: bool = False
    ):
        for i, value in enumerate(tp):
            if isinstance(value, tuple):
                cur_prefix = "" if (rm_zero and i == 0) else f"{prefix}{i}{self._sep}"
                self._process_auto(value, cur_prefix, 0, True if i == 0 else False)
            else:
                for k, v in value.items():
                    cur_prefix = (
                        k if (rm_zero and i == 0) else f"{prefix}{i}{self._last_sep}{k}"
                    )
                    self._tuple_dict[cur_prefix] = v

    def _process_depth(
        self,
        tp: Union[Tuple[Item], Item],
        prefix: str,
        depth: int,
        rm_zero: bool = False,
    ):
        if depth > 1:
            for i, value in enumerate(tp):
                cur_prefix = "" if (rm_zero and i == 0) else f"{prefix}{i}{self._sep}"
                self._process_depth(
                    value, cur_prefix, depth - 1, True if i == 0 else False
                )
        else:
            for i, value in enumerate(tp):
                for k, v in value.items():
                    cur_prefix = (
                        k if (rm_zero and i == 0) else f"{prefix}{i}{self._last_sep}{k}"
                    )
                    self._tuple_dict[cur_prefix] = v


if __name__ == "__main__":
    import time
    import timeit

    print("---- Auto depth ----")

    for tuple_dict in [
        ({"a": 1}, {"b": 2}),
        ({"a": 1}, ({"b": 2}, {"c": 3})),
        ({"a": 1}, ({"b": 2}, ({"c": 3}, {"d": 4}))),
    ]:
        dict_tuple = DictTuple(DictTupleConfig(separator="/"))
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
        dict_tuple = DictTuple(DictTupleConfig(depth=index, separate_key=False))
        start = time.perf_counter()
        result = dict_tuple(tuple_dict)
        print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
        print(result)

    # Benchmark
    print("---- Benchmark ----")
    tuple_dict = (
        ({"a": 1}, {"b": 2}, {"c": 3}),
        ({"d": 4}, {"e": 5}, {"f": 6}),
    )
    dict_tuple_auto = DictTuple(DictTupleConfig(separator="/"))
    dict_tuple_depth = DictTuple(DictTupleConfig(depth=2, separator="/"))
    n_runs = 10000
    time_auto = timeit.timeit(lambda: dict_tuple_auto(tuple_dict), number=n_runs)
    time_depth = timeit.timeit(lambda: dict_tuple_depth(tuple_dict), number=n_runs)
    print(f"Total time for {n_runs} runs (auto depth): {time_auto * 1000:.3f} ms")
    print(f"Total time for {n_runs} runs (with depth): {time_depth * 1000:.3f} ms")
