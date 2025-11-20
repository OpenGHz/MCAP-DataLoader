from pydantic import BaseModel, Field
from collections.abc import Iterator, Callable
from typing import Generic, Union
from mcap_data_loader.pipelines.basis import Pipeline, T
from mcap_data_loader.utils.extra_itertools import recursive_map


class MapConfig(BaseModel, Generic[T], frozen=True):
    """Configuration for Map pipeline."""

    callable: Callable[..., T]
    """Callable to apply to each item at the specified depth."""

    depth: Union[int, None] = Field(default=0, ge=-1)
    """
    Depth level to apply the callable."""


class Map(Pipeline[T]):
    """Map pipeline that applies a callable to items at a given depth in nested iterables."""

    def __init__(self, config: MapConfig[T]) -> None:
        self._callable = config.callable
        self._depth = config.depth

    def __iter__(self) -> Iterator[T]:
        return recursive_map(self._callable, self._iterable, self._depth)


if __name__ == "__main__":
    mapper = Map(MapConfig(callable=lambda x: x * 2, depth=1))

    nested_iterable = [[1, 2], [3, 4], [5, 6]]
    mapper(nested_iterable)
    for item in mapper:
        print(list(item))
