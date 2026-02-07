from pydantic import BaseModel, NonNegativeInt
from collections.abc import Mapping, Callable, Iterable
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.utils.extra_itertools import recursive_map_reusable
from mcap_data_loader.utils.dict import valmap_depth
from mcap_data_loader.basis import StrEnum
from enum import auto
from typing import Literal, Optional, Union


class MappingStrategy(StrEnum):
    """Strategy for the Map caller when the input data is a mapping."""

    FORBID = auto()
    """Forbid the input data to be a mapping. An error will be raised if the input data is a mapping."""
    PASS = auto()
    """Pass the input data directly if it is a mapping."""
    KEY = auto()
    """Apply the callable to the keys of the mapping if the input data is a mapping."""
    VALUE = auto()
    """Apply the callable to the values of the mapping if the input data is a mapping."""
    ITEM = auto()
    """Apply the callable to the items of the mapping if the input data is a mapping."""


class DictApplyConfig(BaseModel, frozen=True):
    """Configuration for applying a callable to dict data."""

    """The callable to apply to the dict data."""
    strategy: MappingStrategy = MappingStrategy.VALUE
    """Strategy for the Map caller when the input data is a dict."""
    depth: int = 0
    """The depth to apply the callable to the dict data."""
    # include: Set[str] = set()
    # """The keys to include when applying the callable to the dict data. If empty, all keys will be included."""
    # exclude: Set[str] = set()
    # """The keys to exclude when applying the callable to the dict data. If empty, no keys will be excluded."""


class DictStrategyMapConfig(DictApplyConfig):
    """Configuration for Map caller when the input data is a dict."""

    callable: Callable


class DictStrategyMap(CallerBasis[Union[Mapping, Iterable]]):
    """A caller that applies the given callable to the input dict data according to the given strategy"""

    def __init__(self, config: DictStrategyMapConfig):
        self.config = config
        self._depth = config.depth
        self._call = config.callable
        self._strategy = config.strategy
        self._strategies = {
            MappingStrategy.FORBID: self._forbid,
            MappingStrategy.PASS: lambda data: data,
            MappingStrategy.VALUE: self._value,
            MappingStrategy.KEY: self._key,
            MappingStrategy.ITEM: self._item,
        }

    def _forbid(self, data):
        raise ValueError("Input data is a dict, but strategy is FORBID.")

    def _value(self, data: Mapping):
        return valmap_depth(self._call, data, self._depth)

    def _item(self, data: Mapping):
        return {k: v for k, v in self._reusable_map(data.items(), self._depth)}

    def _key(self, data: Mapping):
        return {self._call(k): v for k, v in data.items()}

    def _reusable_map(self, data: Iterable, depth: int = 0) -> Iterable:
        return recursive_map_reusable(self._call, data, depth)

    def __call__(self, data: Mapping):
        # bad performance, so we don't support include and exclude for now
        # `valmap_include` can be used instead
        # self._keys = (self.config.include or data.keys()) - self.config.exclude
        return self._strategies[self._strategy](data)


class MustConfig(BaseModel, frozen=True):
    """Configuration for Must checker."""

    mapping: bool = True
    """Whether the input data must be a mapping or must not be a mapping."""
    mode: Literal["pass", "forbid", "direct"] = "direct"
    """Strategy for the must checker. If 'pass', the input data will be directly returned. 
    If 'forbid', an error will be raised if type mismatched. If 'direct', the input data will be processed
    directly without any divergence."""


class MapConfig(BaseModel, frozen=True):
    """Configuration for Map caller."""

    nested: NonNegativeInt = 0
    """The number of nested Map callers to apply."""
    depth: int = 0
    """The depth to diverge the input data."""
    callable: Callable
    """The callable to apply to each diverged branch."""
    mapping: MappingStrategy = MappingStrategy.VALUE
    """Strategy to apply the callable when the input data is a mapping."""
    must: Optional[MustConfig] = None
    """Configuration for the must checker. A mapping is a special kind of iterable because we usually don't want to process it by direct iteration, since direct iteration returns the key. If None, no check will be performed, and
    the data will be processed by the mapping strategy when it is a mapping or by direct iteration when it is not a mapping."""
    # slicing: Optional[SliceConfig] = None
    # """Only apply the callable to the specified slice of the data
    # and keep the rest unchanged."""


class Map(CallerBasis):
    """A caller that diverges the input data into multiple branches based on the given depth."""

    def __init__(self, config: MapConfig):
        self._nested = config.nested
        self._depth = config.depth
        self._callable = config.callable
        self._dict_mapper = DictStrategyMap(
            DictStrategyMapConfig(
                callable=self._callable,
                strategy=config.mapping,
                depth=self._depth,
            )
        )
        self.config = config
        must = {}
        if config.must is not None:
            must[config.must.mapping] = config.must.mode
        # print(must)
        self._must = must

    def _recur_map(self, data: Iterable) -> Iterable:
        return recursive_map_reusable(self._callable, data, self._depth)

    def _call(self, data):
        return self._must_check(isinstance(data, Mapping), data)

    def __call__(self, data: Iterable) -> Iterable:
        if self._nested > 0:
            return recursive_map_reusable(self._call, data, self._nested - 1)
        else:
            return self._call(data)

    def _must_check(self, is_mapping: bool, data):
        # if the data is a mapping, we get from False key from the must config
        # and when must mapping is also False, i.e. the data should be checked
        # when the data is not a mapping, the value will not be None and thus
        # we can check with this config
        mode = self._must.get(not is_mapping)
        if mode is None:  # no need to check
            if isinstance(data, Mapping):
                return self._dict_mapper(data)
            else:
                return self._recur_map(data)
        else:
            if mode == "forbid":
                raise ValueError(f"Input data must not be a {is_mapping}.")
            elif mode == "pass":
                return data
            else:  # direct
                return self._callable(data)


if __name__ == "__main__":
    from pprint import pprint
    from more_itertools import collapse

    dict_mapper = DictStrategyMap(
        DictStrategyMapConfig(
            callable=lambda x: x + 1,
            strategy=MappingStrategy.VALUE,
        )
    )
    data = {"a": 0, "b": 1}
    result = dict_mapper(data)
    assert result == {"a": 1, "b": 2}

    mapper = Map(MapConfig(depth=0, callable=lambda x: x + 1))

    data = {"a": 0, "b": 1}
    result = mapper(data)
    assert result == {"a": 1, "b": 2}

    mapper = Map(MapConfig(depth=1, callable=lambda x: x + 1))

    data = [[1, 2], [3, 4], [5, 6]]
    result = mapper(data)
    assert list(collapse(result)) == [2, 3, 4, 5, 6, 7]
    assert list(collapse(result)) == [2, 3, 4, 5, 6, 7]

    mapper = Map(MapConfig(depth=1, callable=lambda x: x * 2, must=MustConfig()))
    assert mapper([1, 2, 3]) == [1, 2, 3, 1, 2, 3]

    mapper = Map(
        MapConfig(nested=1, callable=lambda v: v + 1, mapping=MappingStrategy.VALUE)
    )
    data = [{"a": 0, "b": 1}]
    result = mapper(data)
    for item in result:
        assert item == {"a": 1, "b": 2}, item
