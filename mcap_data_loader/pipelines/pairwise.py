from mcap_data_loader.utils.extra_itertools import epairwise, Reusablizer
from pydantic import BaseModel, NonNegativeInt, Field
from mcap_data_loader.pipelines.basis import Pipe, T
from typing import Any, Tuple
from collections.abc import Iterable


class PairWiseConfig(BaseModel, frozen=True):
    """Configuration for PairWise pipeline."""

    gap: NonNegativeInt = 0
    """Number of elements to skip between pairs."""
    fillvalue: Any = Field(default_factory=lambda: ...)
    """Value to use for filling missing elements."""
    fill_with_last: bool = False
    """Whether to fill missing elements with the last element."""

    @property
    def future_span(self) -> int:
        """Calculate the number of times the last item leads the current item.
        If the current item index is `t`, then `t + future_span` points exactly
        to the last future item."""
        return self.gap + 1


class PairWise(Pipe[T]):
    def __init__(self, config: PairWiseConfig) -> None:
        self._config_dict = config.model_dump()
        self._epairwise = Reusablizer(epairwise)
        # Set future_span attribute to be used by Slice
        setattr(self._epairwise, "future_span", config.future_span)

    def on_call(self, iterable: Iterable[T]) -> Reusablizer[Tuple[T, T]]:
        return self._epairwise(iterable, **self._config_dict)


if __name__ == "__main__":
    iterable = range(5)

    config = PairWiseConfig(gap=10, fill_with_last=True)
    paired = PairWise(config)(iterable)

    for item in paired:
        print(item)
    assert list(paired)
