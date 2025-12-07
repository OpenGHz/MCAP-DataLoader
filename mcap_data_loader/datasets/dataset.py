import random
from typing import Any, List, Optional, Generic, TypeVar, Protocol, final
from typing_extensions import Self, runtime_checkable
from collections.abc import Iterator, Iterable, Generator
from pydantic import BaseModel, ConfigDict, computed_field
from abc import abstractmethod
from functools import cached_property, cache
from logging import getLogger
from mcap_data_loader.utils.basic import (
    StrEnum,
    multi_slices_to_indexes,
    DictableSlicesType,
    DictableIndexesType,
)
from mcap_data_loader.basis.cfgable import InitConfigABCMixin
from enum import auto
from more_itertools import nth, ilen
from pathlib import Path
from natsort import natsort_keygen


T = TypeVar("T")
MultiSampleT = TypeVar("MultiSampleT", bound=Iterable)
MultiEpisodeT = TypeVar("MultiEpisodeT", bound=Iterable[Iterable])


class RearrangeType(StrEnum):
    NONE = auto()
    """ No rearrangement."""
    SORT = auto()
    """ Sort the data in ascending order."""
    SORT_STEM_DIGITAL = auto()
    """ Sort the data by the numeric value of the stem (filename without extension)."""
    NATSORT = auto()
    """ Sort the data using natural order (e.g., '2' before '10')."""
    SHUFFLE = auto()
    """ Shuffle the data randomly."""
    REVERSE = auto()
    """ Reverse the order of the data."""

    @staticmethod
    def rearrange(
        data: List[Any],
        strategy: Self,
        random_generator: Optional[random.Random] = None,
    ) -> None:
        """
        Rearrange the data based on the specified strategy and random generator.
        Args:
            data (List[Any]): The data to rearrange.
            strategy (RearrangeType): The rearrangement strategy to apply.
            random_generator (Optional[random.Random]): Optional random generator for shuffling.
        Raises:
            ValueError: If an unsupported rearrangement strategy is provided.
        """
        if strategy is RearrangeType.SORT:
            data.sort()
        elif strategy is RearrangeType.SORT_STEM_DIGITAL:
            data.sort(key=lambda p: int(p.stem))
        elif strategy is RearrangeType.NATSORT:
            data.sort(key=natsort_keygen())
        elif strategy is RearrangeType.REVERSE:
            data.reverse()
        elif strategy is RearrangeType.SHUFFLE:
            if random_generator is None:
                random.shuffle(data)
            else:
                random_generator.shuffle(data)
        elif strategy is not RearrangeType.NONE:
            raise ValueError(f"Unsupported rearrangement strategy: {strategy}")


class DataSlicesConfig(BaseModel, frozen=True):
    """Configuration for slicing data.
    This class defines how to slice samples, episodes, and datasets.
    Args:
        sample: Consider a flattened dict sample {'key1': [1, 2, 3], 'key2': [4, 5, 6]},
        given the dict slices:  {'key1': (0, 2), 'key2': (1, 3)}, the result will be:
        {'key1': [1, 2], 'key2': [5, 6]}.
        episode: Consider a flattened dataset: {'/path1/episode0': [point1, point2, point3],
        '/path2/episode1': [point1, point2, point3]}, given the dict slices: {'/path1/episode0': (0, 2),
        '/path2/episode1': (1, 3)}, the result will be {'/path1/episode0': [point1, point2],
        '/path2/episode1': [point2, point3]}
        dataset: Consider a flattened dataset with multiple sub-datasets:
        {'dataset1': ['episode1', 'episode2', 'episode3'], 'dataset2': ['episode1', 'episode2', 'episode3']},
        given the dict slices: {'dataset1': (0, 2), 'dataset2': (1, 3)}, the result will be:
        {'dataset1': ['episode1', 'episode2'], 'dataset2': ['episode2', 'episode3']}
    """

    sample: DictableSlicesType = {}
    episode: DictableSlicesType = {}
    dataset: DictableSlicesType = {}

    @staticmethod
    def _slices_to_indexes(slices: DictableSlicesType) -> DictableIndexesType:
        """
        Convert slices to indexes.
        If slices is a dict, convert each key's slices to indexes.
        If slices is a list, convert the list of slices to indexes.
        """
        if isinstance(slices, dict):
            return {k: multi_slices_to_indexes(v) for k, v in slices.items()}
        elif isinstance(slices, list):
            return multi_slices_to_indexes(slices)

    @computed_field
    @cached_property
    def sample_indexes(self) -> DictableIndexesType:
        return self._slices_to_indexes(self.sample)

    @computed_field
    @cached_property
    def episode_indexes(self) -> DictableIndexesType:
        return self._slices_to_indexes(self.episode)

    @computed_field
    @cached_property
    def dataset_indexes(self) -> DictableIndexesType:
        return self._slices_to_indexes(self.dataset)


class DataRearrangeConfig(BaseModel, frozen=True):
    """Configuration for rearranging data.
    This class defines how to rearrange samples, episodes, and datasets.
    """

    sample: RearrangeType = RearrangeType.NONE
    """Rearrangement strategy for each sample (rarely used)."""
    episode: RearrangeType = RearrangeType.NONE
    """Rearrangement strategy for each episode (e.g. reverse a trajectory)."""
    dataset: RearrangeType = RearrangeType.NONE
    """Rearrangement strategy for the dataset (e.g. sort episodes)."""
    seed: Optional[int] = None
    """Random seed for shuffling, if applicable."""


class IterableDatasetConfig(BaseModel, frozen=True):
    """Iterable Dataset configuration basis."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    data_root: Path
    """Root directory of the dataset."""
    slices: DataSlicesConfig = DataSlicesConfig()
    """Slicing configuration for samples, episodes and datasets"""
    rearrange: DataRearrangeConfig = DataRearrangeConfig()
    """Rearrangement strategy for samples, episodes and datasets."""


@runtime_checkable
class IterableDatasetProtocol(Protocol[T]):
    """
    Protocol for iterable datasets.
    Subclasses only need to implement `__iter__()` to generate samples.
    """

    def __iter__(self) -> Iterator[T]:
        """Return an iterator of samples or episodes."""


@runtime_checkable
class IterableMultiEpisodeDatasetsProtocol(Protocol[MultiEpisodeT]):
    """
    Protocol for iterable multi-episode datasets.
    Subclasses only need to implement `__iter__()` to generate multiple datasets.
    """

    def __iter__(self) -> Iterator[MultiEpisodeT]:
        """Return an iterator of episodes"""


class IterableDatasetABC(InitConfigABCMixin, Generic[T]):
    """
    Generic iterable dataset template.
    Subclasses only need to implement `read_stream()` to generate samples.
    """

    @abstractmethod
    def read_stream(self) -> Iterator[T]:
        """Returns an **iterator object**, each element is a stream item."""
        raise NotImplementedError

    def get_logger(self):
        return getLogger(self.__class__.__name__)

    def close(self) -> None:
        """Close the dataset and release any resources if needed."""

    def statistics(self) -> dict:
        """Get dataset statistics if available."""
        raise NotImplementedError(
            f"Statistics method not implemented for {self.__class__.__name__}."
        )

    def __getitem__(self, index: int) -> T:
        """
        Get a specific sample by index.
        This is not efficient for large datasets, use with caution.
        """
        return nth(self.read_stream(), index)

    @cache
    def __len__(self) -> int:
        """
        Get the total number of samples.
        This is not efficient for large datasets for the first time, use with caution.
        """
        return ilen(self.read_stream())

    @final
    def __iter__(self) -> Iterator[T]:
        return self.read_stream()

    def __del__(self):
        self.close()


class IterableWritableDatasetABC(IterableDatasetABC[T]):
    """Generic iterable writable dataset template.
    Subclasses need to implement both `read_stream()` and `write()` methods.
    """

    @abstractmethod
    def write(self, *args, **kwargs) -> Any:
        """Write anything to the dataset. For example, it can be used to insert,
        append, and extend data into a data stream, or conversely, delete data,
        or dynamically adjust the dataset configuration. Furthermore, this method
        is essential for real-time data streams that require external control updates,
        such as multimodal sensor data from robotic devices. However, in most cases,
        the dataset is static and read-only. Depending on the actual needs, this method can
        accept arbitrary parameters and return any type of values."""


class IterableReadableDatasetABC(IterableDatasetABC[T]):
    """Generic iterable readable dataset template.
    Subclasses need to implement both `read_stream()` and `read()` methods.
    """

    @abstractmethod
    def read(self) -> T:
        """Read a single item from the dataset stream.
        Typically, when a dataset iteration begins, the `read_stream` method is called to return a refreshed iterator. Since the dataset itself cannot record the iterator's state, the `read` method is meaningless. However, in some cases, the data stream is not refreshed by the `read_stream` method, such as a real-time video stream from an external camera. In this case, `read_stream` can be viewed as a loop of calls to the `read` method, which can then be conveniently used to retrieve the latest frame of data.
        """
        raise NotImplementedError("Read method not implemented for this dataset.")


class IterableRWDatasetABC(
    IterableReadableDatasetABC[T], IterableWritableDatasetABC[T]
):
    """Generic iterable readable and writable dataset template.
    Subclasses need to implement `read_stream()`, `read()`, and `write()` methods.
    """


class RealTimeDatasetABC(IterableRWDatasetABC[T]):
    """Generic real-time dataset template.
    Subclasses need to implement `read()`, `write()` and `reset()` methods.
    The real-time dataset is designed for scenarios where data is continuously
    generated and needs to be processed in real-time, such as sensor data from
    robots.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the dataset to its initial state.
        This method is useful in real-time datasets to clear any buffered data
        and prepare for a new data stream.
        """

    @final
    def read_stream(self) -> Generator[T]:
        """Generate an infinite stream of data by repeatedly
        calling the `read` method."""
        while True:
            yield self.read()


if __name__ == "__main__":
    import time

    class TimeReadDataset(IterableReadableDatasetABC[float]):
        config: ...

        def read_stream(self):
            while True:
                yield self.read()

        def read(self):
            return time.time()

    dataset = TimeReadDataset()
    print("Read:", dataset.read())

    class LengthRWDataset(RealTimeDatasetABC[int]):
        def __init__(self, config: ...):
            self.reset()

        def reset(self):
            self._buffer = []

        def read(self):
            return len(self._buffer)

        def write(self, item: int) -> bool:
            self._buffer.append(item)
            return True

    dataset = LengthRWDataset()
    print("Length before write:", dataset.read())
    dataset.write(42)
    print("Length after write:", dataset.read())

    assert isinstance(dataset, IterableDatasetProtocol)
    assert isinstance(dataset, IterableDatasetABC)
    assert isinstance(dataset, IterableRWDatasetABC)
    assert isinstance(dataset, IterableReadableDatasetABC)
    assert isinstance(dataset, IterableWritableDatasetABC)
    assert isinstance(dataset, RealTimeDatasetABC)
