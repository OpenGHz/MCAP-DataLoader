from pydantic import BaseModel, NonNegativeInt
from collections.abc import Mapping, Callable, Iterable
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.utils.extra_itertools import recursive_map_reusable
from mcap_data_loader.utils.dict import valmap_depth
from mcap_data_loader.utils.basic import StrEnum
from enum import auto


class MappingStrategy(StrEnum):
    FORBID = auto()
    PASS = auto()
    KEY = auto()
    VALUE = auto()
    ITEM = auto()


class MapConfig(BaseModel, frozen=True):
    """Configuration for Map caller."""

    depth: NonNegativeInt = 0
    """The depth to diverge the input data."""
    callable: Callable
    """The callable to apply to each diverged branch."""
    mapping: MappingStrategy = MappingStrategy.VALUE
    """Strategy for the diverter caller when the input data is a mapping."""


class Map(CallerBasis):
    """A caller that diverges the input data into multiple branches based on the given depth."""

    def __init__(self, config: MapConfig):
        self._depth = config.depth
        self._callable = config.callable
        self._mapping = config.mapping
        # TODO: support depth for KEY and ITEM strategies
        self._strategies = {
            MappingStrategy.FORBID: self._forbid,
            MappingStrategy.PASS: lambda data: data,
            MappingStrategy.VALUE: self._value,
            MappingStrategy.KEY: self._key,
            MappingStrategy.ITEM: self._item,
        }

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
        if isinstance(data, Mapping):
            return self._strategies[self._mapping](data)
        else:
            return self._recur_map(data)


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
