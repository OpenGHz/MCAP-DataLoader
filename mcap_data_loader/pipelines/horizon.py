from mcap_data_loader.utils.extra_itertools import past_future
from pydantic import BaseModel, NonNegativeInt
from mcap_data_loader.pipelines.basis import Pipeline, T
from typing import Any, Tuple
from collections.abc import Iterator


class HorizonConfig(BaseModel):
    """Configuration for the Horizon pipeline."""

    past_num: NonNegativeInt = 0
    """Number of past items to include in each output tuple."""
    future_num: NonNegativeInt = 0
    """Number of future items to include in each output tuple."""
    fillvalue: Any = None
    """Value to use for filling in missing items."""
    step: NonNegativeInt = 1
    """Step size for sliding window."""
    fill_with_last: bool = False
    """Whether to fill missing items with the last available value."""
    gap: NonNegativeInt = 0
    """Number of items to skip between the last item in the past tuple 
    (i.e. the current item) and the first item in the future. E.g. 0 means
    the current item is included in the future."""


class Horizon(Pipeline[T]):
    def __init__(self, config: HorizonConfig) -> None:
        self.config_dict = config.model_dump()

    def __iter__(self) -> Iterator[Tuple[Tuple[T, ...], Tuple[T, ...]]]:
        return past_future(self._iterable, **self.config_dict)


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
