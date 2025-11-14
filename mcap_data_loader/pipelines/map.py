from pydantic import BaseModel
from collections.abc import Iterator
from typing import Generic
from mcap_data_loader.pipelines.basis import Pipeline, T
from collections.abc import Callable


class MapConfig(BaseModel, Generic[T]):
    """Configuration for Map pipeline."""

    callable: Callable[..., T]
    """Callable to apply to each item in the iterable."""


class Map(Pipeline[T]):
    """Map pipeline that applies a callable to each item in the iterable."""

    def __init__(self, config: MapConfig[T]) -> None:
        self.config = config

    def __iter__(self) -> Iterator[T]:
        # TODO: should we reset the callable if it has a reset method?
        # if hasattr(self.config.callable, "reset"):
        #     self.config.callable.reset()
        return map(self.config.callable, self._iterable)


if __name__ == "__main__":

    def square(x) -> int:
        return x * 2

    numbers = [1, 2, 3, 4, 5]
    map_iterable = Map(MapConfig(callable=square))(numbers)

    for item in map_iterable:
        print(item)
