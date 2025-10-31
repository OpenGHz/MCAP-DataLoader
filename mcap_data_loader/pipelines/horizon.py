from mcap_data_loader.utils.extra_itertools import past_future
from pydantic import BaseModel, NonNegativeInt
from mcap_data_loader.pipelines.basis import Pipeline, T
from typing import Any, Tuple
from collections.abc import Generator


class HorizonConfig(BaseModel):
    past_num: NonNegativeInt = 0
    future_num: NonNegativeInt = 0
    fillvalue: Any = None
    step: NonNegativeInt = 1
    fill_with_last: bool = False


class Horizon(Pipeline[T]):
    def __init__(self, config: HorizonConfig) -> None:
        self.config = config

    def __iter__(self) -> Generator[Tuple[Tuple[T, ...], Tuple[T, ...]]]:
        config = self.config
        yield from past_future(
            self._iterable,
            config.past_num,
            config.future_num,
            config.fillvalue,
            config.step,
            config.fill_with_last,
        )


if __name__ == "__main__":
    iterable = range(5)

    config = HorizonConfig(past_num=1, future_num=2, fill_with_last=True)
    past_futured = Horizon(config)(iterable)

    for item in past_futured:
        print(item)
