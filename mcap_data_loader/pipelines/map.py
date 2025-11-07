from pydantic import BaseModel
from collections.abc import Generator
from typing import Generic
from mcap_data_loader.pipelines.basis import Pipeline, T
from collections.abc import Callable


class MapConfig(BaseModel, Generic[T]):
    callable: Callable[..., T]


class Map(Pipeline[T]):
    """Map pipeline that applies a callable to each item in the iterable."""

    def __init__(self, config: MapConfig[T]) -> None:
        self.config = config

    def __iter__(self) -> Generator[T]:
        yield from map(self.config.callable, self._iterable)


if __name__ == "__main__":

    def square(x) -> int:
        return x * 2

    numbers = [1, 2, 3, 4, 5]
    map_iterable = Map(MapConfig(callable=square))(numbers)

    for item in map_iterable:
        print(item)
