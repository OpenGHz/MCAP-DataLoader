from pydantic import BaseModel
from collections.abc import Mapping, Callable, Iterable
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.utils.extra_itertools import recursive_map_reusable
from mcap_data_loader.utils.dict import valmap_depth
from mcap_data_loader.utils.basic import StrEnum
from enum import auto
from typing import Literal, Optional


class MappingStrategy(StrEnum):
    FORBID = auto()
    PASS = auto()
    KEY = auto()
    VALUE = auto()
    ITEM = auto()


class MustConfig(BaseModel):
    """Configuration for Must caller."""

    mapping: bool = True
    """Whether the input data must be a mapping or must not be a mapping."""
    mode: Literal["pass", "forbid", "direct"] = "direct"
    """Strategy for the must checker. If 'pass', the input data will be directly returned. 
    If 'forbid', an error will be raised. If 'direct', the input data will be processed
    directly without any divergence."""


class MapConfig(BaseModel, frozen=True):
    """Configuration for Map caller."""

    depth: int = 0
    """The depth to diverge the input data."""
    callable: Callable
    """The callable to apply to each diverged branch."""
    mapping: MappingStrategy = MappingStrategy.VALUE
    """Strategy for the diverter caller when the input data is a mapping."""
    must: Optional[MustConfig] = None
    """Configuration for the must checker."""
    # slicing: Optional[SliceConfig] = None
    # """Only apply the callable to the specified slice of the data
    # and keep the rest unchanged."""


class Map(CallerBasis):
    """A caller that diverges the input data into multiple branches based on the given depth."""

    def __init__(self, config: MapConfig):
        self._depth = config.depth
        self._callable = config.callable
        self._mapping = config.mapping
        # self._slicing = config.slicing
        # TODO: support depth for KEY and ITEM strategies
        self._strategies = {
            MappingStrategy.FORBID: self._forbid,
            MappingStrategy.PASS: lambda data: data,
            MappingStrategy.VALUE: self._value,
            MappingStrategy.KEY: self._key,
            MappingStrategy.ITEM: self._item,
        }
        self.config = config
        must = {}
        if config.must is not None:
            if config.must.mapping:
                must["mapping"] = config.must.mode
            else:
                must["not_mapping"] = config.must.mode
        # print(must)
        self._must = must

    def _forbid(self, data):
        raise ValueError("Input data is a mapping, but mapping strategy is FORBID.")

    def _value(self, data: Mapping):
        return valmap_depth(self._callable, data, self._depth)

    def _item(self, data: Mapping):
        return self._recur_map(data.items())

    def _key(self, data: Mapping):
        return self._recur_map(data.keys())

    def _recur_map(self, data: Iterable) -> Iterable:
        return recursive_map_reusable(self._callable, data, self._depth)

    def __call__(self, data: Iterable) -> Iterable:
        return self._must_check(
            "not_mapping" if isinstance(data, Mapping) else "mapping", data
        )

    def _must_check(self, name: str, data):
        mode = self._must.get(name, None)
        if mode is None:
            if isinstance(data, Mapping):
                return self._strategies[self._mapping](data)
            else:
                return self._recur_map(data)
        else:
            if mode == "forbid":
                raise ValueError(f"Input data must not be a {name}.")
            elif mode == "pass":
                return data
            else:  # direct
                return self._callable(data)


if __name__ == "__main__":
    from pprint import pprint
    from more_itertools import collapse

    mapper = Map(MapConfig(depth=0, callable=lambda x: x + 1))

    data = {"a": 0, "b": 1}
    result = mapper(data)
    pprint(result)

    mapper = Map(MapConfig(depth=1, callable=lambda x: x + 1))

    data = [[1, 2], [3, 4], [5, 6]]
    result = mapper(data)
    assert list(collapse(result)) == [2, 3, 4, 5, 6, 7]
    assert list(collapse(result)) == [2, 3, 4, 5, 6, 7]

    mapper = Map(MapConfig(depth=1, callable=lambda x: x * 2, must=MustConfig()))
    assert mapper([1, 2, 3]) == [1, 2, 3, 1, 2, 3]
