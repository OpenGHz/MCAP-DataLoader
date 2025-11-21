"""Slices the data from the start index to the end index with a specified step."""

from collections.abc import Iterable
from typing import Optional
from more_itertools import islice_extended
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from mcap_data_loader.pipelines.basis import Pipe, T
from mcap_data_loader.utils.basic import try_to_get_attr
from mcap_data_loader.utils.extra_itertools import Reusablizer


class SliceConfig(BaseModel, frozen=True):
    """Configuration for the Slice pipeline."""

    start: NonNegativeInt = 0
    """Starting index (inclusive) for the slice."""

    stop: Optional[NonNegativeInt] = None
    """Stopping index (exclusive) for the slice. ``None`` means go to the end."""

    step: Optional[PositiveInt] = 1
    """Stride of the slice. If ``None``, it will be inferred from the input iterable (future_span)."""


class Slice(Pipe):
    """Yield items from the iterable according to the configured slice."""

    def __init__(self, config: SliceConfig) -> None:
        self.config = config
        self._islice = Reusablizer(islice_extended)
        self._ss = (
            config.start == 0
            and config.stop is None
            and (config.step == 1 or config.step is None)
        )

    def on_call(self, iterable: Iterable[T]) -> Iterable[T]:
        sss = self._ss
        if self.config.step is None:
            step = try_to_get_attr(iterable, ["config.future_span", "future_span"])
            sss *= step == 1
        else:
            step = self.config.step
        if sss:  # No slicing needed
            return iterable
        return self._islice(iterable, self.config.start, self.config.stop, step)


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
