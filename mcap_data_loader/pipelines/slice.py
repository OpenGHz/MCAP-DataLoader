"""Slices the data from the start index to the end index with a specified step."""

from collections.abc import Iterator
from typing import Optional
from more_itertools import islice_extended
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from mcap_data_loader.pipelines.basis import Pipe, T
from mcap_data_loader.utils.basic import try_to_get_attr


class SliceConfig(BaseModel, frozen=True):
    """Configuration for the Slice pipeline."""

    start: NonNegativeInt = 0
    """Starting index (inclusive) for the slice."""

    stop: Optional[NonNegativeInt] = None
    """Stopping index (exclusive) for the slice. ``None`` means go to the end."""

    step: Optional[PositiveInt] = 1
    """Stride of the slice. If ``None``, it will be inferred from the input iterable (future_span)."""


class Slice(Pipe[T]):
    """Yield items from the iterable according to the configured slice."""

    def __init__(self, config: SliceConfig) -> None:
        self.config = config

    def on_call(self, iterable):
        if self.config.step is None:
            step = try_to_get_attr(iterable, ["config.future_span", "future_span"])
        else:
            step = self.config.step
        self._step = step
        return super().on_call(iterable)

    def __iter__(self) -> Iterator[T]:
        return islice_extended(
            self._iterable, self.config.start, self.config.stop, self._step
        )


__all__ = ["Slice", "SliceConfig"]


if __name__ == "__main__":

    class Iterable:
        class config:
            future_span = 2

        def __iter__(self):
            return iter(range(10))

    iterable = Iterable()
    args = ((0, None, 1), (0, None, 2), (2, 8, 2), (2, 8, None))
    expected = [
        tuple(iterable),
        tuple(range(0, 10, 2)),
        (2, 4, 6),
        (2, 4, 6),
    ]
    for arg, exp in zip(args, expected):
        config = SliceConfig(start=arg[0], stop=arg[1], step=arg[2])
        sliced = Slice(config)(iterable)
        assert tuple(sliced) == exp
        print("Sliced output:", tuple(sliced))
