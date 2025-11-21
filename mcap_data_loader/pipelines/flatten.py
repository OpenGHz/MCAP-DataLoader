from pydantic import BaseModel
from collections.abc import Generator, Iterator
from mcap_data_loader.pipelines.basis import Pipe, T
from typing import Optional, Tuple, Union
from more_itertools import collapse


class FlattenConfig(BaseModel, frozen=True):
    """Configuration for Flatten pipeline."""

    depth: int = -1
    """Depth to which to flatten nested iterables.
    A depth of 0 means no flattening. Negative values mean flatten all levels
    (much slower but useful for non-single nested levels). Positive values mean
    only flatten up to that depth."""
    base_type: Optional[Union[Tuple[type, ...], type]] = None
    """Binary and text strings are not considered iterable and
    will not be collapsed. To avoid collapsing other types, specify it here.
    This is only used when depth is negative."""


class Flatten(Pipe[T]):
    """Flatten nested iterables up to a specified depth."""

    def __init__(self, config: FlattenConfig) -> None:
        self._depth = config.depth
        self._base_type = config.base_type

    def __iter__(self) -> Iterator[T]:
        if self._depth < 0:
            yield from collapse(self._iterable, base_type=self._base_type)
        else:
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
    import time

    nested_list = [[1, 2], [3, [4, 5]], ["123", {1: 2}]]

    for depth in (-1, 0, 1):
        flattene = Flatten(FlattenConfig(depth=depth, base_type=dict))
        start = time.perf_counter()
        flat = flattene(nested_list)
        print(f"{depth=}", tuple(flat), time.perf_counter() - start)
