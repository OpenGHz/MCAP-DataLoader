from mcap_data_loader.utils.extra_itertools import epairwise
from pydantic import BaseModel, NonNegativeInt, Field
from mcap_data_loader.pipelines.basis import Pipeline, T
from typing import Any, Tuple
from collections.abc import Generator


class PairWiseConfig(BaseModel):
    gap: NonNegativeInt = 0
    fillvalue: Any = Field(default_factory=lambda: ...)
    fill_with_last: bool = False


class PairWise(Pipeline[T]):
    def __init__(self, config: PairWiseConfig) -> None:
        self.config = config

    def __iter__(self) -> Generator[Tuple[T, T]]:
        config = self.config
        yield from epairwise(
            self._iterable, config.gap, config.fillvalue, config.fill_with_last
        )


if __name__ == "__main__":
    iterable = range(5)

    config = PairWiseConfig(gap=1, fill_with_last=True)
    paired = PairWise(config)(iterable)

    for item in paired:
        print(item)
