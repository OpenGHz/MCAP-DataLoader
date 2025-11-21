from mcap_data_loader.utils.extra_itertools import past_future, Reusablizer
from mcap_data_loader.pipelines.basis import Pipe, T
from pydantic import BaseModel, NonNegativeInt
from typing import Any, Tuple
from collections.abc import Iterable


class HorizonConfig(BaseModel, frozen=True):
    """Configuration for the Horizon pipeline."""

    past_num: NonNegativeInt = 0
    """Number of past items to be included in the first (past) tuple.
    The length of the past tuple will be `past_num + 1`, as the current item
    is always the last item."""
    future_num: NonNegativeInt = 0
    """Number of future items to be included in the second (future) tuple.
    The length of the future tuple will be `future_num + 1`."""
    fillvalue: Any = None
    """Value to use for filling in missing items."""
    step: NonNegativeInt = 1
    """Step between items."""
    fill_with_last: bool = False
    """Whether to fill missing items with the last available value."""
    gap: NonNegativeInt = 0
    """Number of items to skip between the last item in the past tuple 
    (i.e. the current item) and the first item in the future. E.g. 0 means
    the first item is the current item."""

    @property
    def future_span(self) -> int:
        """Calculate the number of times the last item leads the current item.
        If the current item index is `t`, then `t + future_span` points exactly
        to the last future item."""
        return self.gap + self.future_num


class Horizon(Pipe):
    def __init__(self, config: HorizonConfig) -> None:
        self.config_dict = config.model_dump()
        self._past_future = Reusablizer(past_future)
        setattr(self._past_future, "future_span", config.future_span)

    def on_call(
        self, iterable: Iterable[T]
    ) -> Iterable[Tuple[Tuple[T, ...], Tuple[T, ...]]]:
        return self._past_future(iterable, **self.config_dict)


if __name__ == "__main__":
    iterable = range(5)
    gap = 2
    for past_num, future_num in [(0, 0), (0, 2), (2, 0), (1, 2), (2, 2)]:
        print(f"\n--- past_num={past_num}, future_num={future_num} ---")
        config = HorizonConfig(
            past_num=past_num, future_num=future_num, fill_with_last=True, gap=gap
        )
        past_futured = Horizon(config)(iterable)

        for item in past_futured:
            print(item)
