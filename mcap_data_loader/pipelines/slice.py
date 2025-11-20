"""Slices the data from the start index to the end index with a specified step."""

from collections.abc import Iterator
from typing import Optional
from more_itertools import islice_extended
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from mcap_data_loader.pipelines.basis import Pipeline, T


class SliceConfig(BaseModel, frozen=True):
    """Configuration for the Slice pipeline."""

    start: NonNegativeInt = 0
    """Starting index (inclusive) for the slice."""

    stop: Optional[NonNegativeInt] = None
    """Stopping index (exclusive) for the slice. ``None`` means go to the end."""

    step: PositiveInt = 1
    """Stride of the slice."""


class Slice(Pipeline[T]):
    """Yield items from the iterable according to the configured slice."""

    def __init__(self, config: SliceConfig) -> None:
        self._config_dict = config.model_dump()

    def __iter__(self) -> Iterator[T]:
        return islice_extended(self._iterable, **self._config_dict)


__all__ = ["Slice", "SliceConfig"]


if __name__ == "__main__":
    iterable = range(10)

    args = ((0, None, 1), (0, None, 2), (2, 8, 2))
    expected = [
        tuple(iterable),
        tuple(range(0, 10, 2)),
        (2, 4, 6),
    ]
    for arg, exp in zip(args, expected):
        config = SliceConfig(start=arg[0], stop=arg[1], step=arg[2])
        sliced = Slice(config)(iterable)
        assert tuple(sliced) == exp
        print("Sliced output:", tuple(sliced))
