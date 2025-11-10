from pydantic import BaseModel, NonNegativeInt
from collections.abc import Generator
from mcap_data_loader.pipelines.basis import Pipeline, T


class FlattenConfig(BaseModel):
    depth: NonNegativeInt = 1


class Flatten(Pipeline[T]):
    """Flatten nested iterables up to a specified depth."""

    def __init__(self, config: FlattenConfig) -> None:
        self.config = config
        self._depth = config.depth

    def __iter__(self) -> Generator[T]:
        for item in self._iterable:
            yield from self._flatten(item, 0)

    def _flatten(self, iterable: T, current_depth: int) -> Generator[T]:
        """Recursively flatten items up to the specified depth."""
        if current_depth < self._depth:
            for sub_item in iterable:
                yield from self._flatten(sub_item, current_depth + 1)
        else:
            yield iterable


if __name__ == "__main__":
    nested_list = [[1, 2], [3, [4, 5]], [7, 8]]
    flatten_iterable = Flatten(FlattenConfig())(nested_list)

    for item in flatten_iterable:
        print(item)
