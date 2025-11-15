from mcap_data_loader.datasets.dataset import IterableWritableDatasetABC
from mcap_data_loader.callers.basis import CallerBasis, ReturnT
from mcap_data_loader.utils.basic import create_sleeper
from pydantic import BaseModel, ConfigDict
from typing import Generic


class DatasetCallerConfig(BaseModel, Generic[ReturnT], frozen=True):
    """Configuration for dataset callers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: IterableWritableDatasetABC[ReturnT]
    """The dataset to be called."""
    rate: int = -1
    """The rate (in Hz) at which to call the dataset. Default is -1 (no rate limiting)."""


class DatasetCaller(CallerBasis[ReturnT]):
    """Caller that retrieves data from an IterableDatasetABC."""

    def __init__(self, config: DatasetCallerConfig[ReturnT]):
        self._dataset = config.dataset
        self._sleeper = create_sleeper(config.rate)
        self.reset()

    def reset(self) -> None:
        """Reset the internal iterator of the dataset."""
        self._iterator = iter(self._dataset)

    def __call__(self, *args, **kwds) -> ReturnT:
        """Retrieve the next item from the dataset.
        If the dataset supports writing, perform a write operation before reading.
        """
        self._sleeper.reset()
        if self._dataset.write(*args, **kwds) is False:
            raise RuntimeError("Dataset write operation failed.")
        self._sleeper.sleep()
        return next(self._iterator)


if __name__ == "__main__":
    from collections.abc import Generator

    class TestDataset(IterableWritableDatasetABC[int]):
        def __init__(self, config):
            self._buffer = []

        def read_stream(self) -> Generator[int]:
            yield from self._buffer

        def write(self, item: int) -> bool:
            self._buffer.append(item)
            return True

    dataset = TestDataset(...)

    caller = DatasetCaller(DatasetCallerConfig(dataset=dataset, rate=-1))

    for i in range(5):
        assert caller(i) == i
