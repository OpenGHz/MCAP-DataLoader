import random
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Dict,
    Union,
)
from pydantic import BaseModel, NonNegativeInt, computed_field
from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from mcap_data_loader.utils.basic import StrEnum, SlicesType, multi_slices_to_indexes
from enum import auto

try:
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError as e:

    class IterableDataset:
        pass

    # Dummy function if torch is not available
    get_worker_info = lambda: None  # noqa: E731
    getLogger(__name__).warning(
        "torch.utils.data is not available, some features may not work. "
        "Please install PyTorch to use these features."
    )

DictableSlicesType = Union[Dict[str, SlicesType], SlicesType]
DictableIndexesType = Union[Dict[str, List[int]], List[int]]


class RearrangeType(StrEnum):
    NONE = auto()
    SORT = auto()
    SHUFFLE = auto()
    REVERSE = auto()


class DataSlicesConfig(BaseModel):
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


class DataRearrangeConfig(BaseModel):
    """Configuration for rearranging data.
    This class defines how to rearrange samples, episodes, and datasets.
    Args:
        sample: Rearrangement strategy for each sample (rarely used).
        episode: Rearrangement strategy for each episode (e.g. reverse a trajectory).
        dataset: Rearrangement strategy for the dataset.
    """

    sample: RearrangeType = RearrangeType.NONE
    episode: RearrangeType = RearrangeType.NONE
    dataset: RearrangeType = RearrangeType.NONE

    @staticmethod
    def rearrange(
        data: List[Any],
        strategy: RearrangeType,
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
        Description:
            - "sort": Sort the data in ascending order.
            - "shuffle": Shuffle the data randomly using the provided random generator.
            - "none": No rearrangement is applied.
        """
        if strategy == RearrangeType.SORT:
            data.sort()
        elif strategy == RearrangeType.SHUFFLE:
            if random_generator is None:
                random.shuffle(data)
            else:
                random_generator.shuffle(data)
        elif strategy != RearrangeType.NONE:
            raise ValueError(f"Unsupported rearrangement strategy: {strategy}")


class IterableDatasetConfig(BaseModel):
    """Generic iterable Dataset configuration.
    Contains data root directory, random seed, multi-process configuration, etc.
    Subclasses can extend this configuration class to add specific parameters.
    Args:
        data_root (str, List[str]): Raw data root directory/file paths
        shuffle_buffer_size (NonNegativeInt): Buffer size for streaming shuffle
        seed (Optional[int]): Random seed; None means not fixed
        world_size (int): Total number of processes (for distributed training)
        rank (int): Current process rank
        resume_from_sample (int): Resume from the Nth sample
        transform (Optional[Callable[[Any], Any]]): Sample-level transform function
        filter_fn (Optional[Callable[[Any], bool]]): Filter function
        slices (DataSlicesConfig): Slicing configuration for samples, episodes, and datasets
        rearrange (Literal["none", "sort", "shuffle"]): Rearrangement strategy for episodes.
            Each dataset is processed separately.
    Description:
        - `data_root` can be file path, URL or other data source prefix
        - `shuffle_buffer_size` of 0 means no shuffle
        - `seed` controls randomness, None means different each run
        - `world_size` and `rank` for distributed training, ensuring each sample is processed only once
        - `resume_from_sample` for checkpoint resumption, starting from specified sample
        - `transform` and `filter_fn` for sample-level transformation and filtering
    """

    data_root: Union[str, List[str]]
    shuffle_buffer_size: NonNegativeInt = 0
    seed: Optional[int] = None
    world_size: NonNegativeInt = 1
    rank: NonNegativeInt = 0
    resume_from_sample: NonNegativeInt = 0
    transform: Optional[Callable[[Any], Any]] = None
    filter_fn: Optional[Callable[[Any], bool]] = None
    slices: DataSlicesConfig = DataSlicesConfig()
    rearrange: DataRearrangeConfig = DataRearrangeConfig()


class IterableDatasetABC(IterableDataset, ABC):
    """
    Generic iterable dataset template.
    Subclasses only need to implement `_read_stream()` to generate samples.
    """

    def __init__(self, config: IterableDatasetConfig) -> None:
        super().__init__()
        self.cfg = config
        self._rng = random.Random(self.cfg.seed)

    def load(self):
        """
        Load the dataset into memory or prepare it for streaming.
        """

    @abstractmethod
    def _read_stream(self) -> Iterable[Any]:
        """
        Returns an **iterable object**, each element is a sample.
        Subclasses read files, databases, network streams, etc. based on data_root.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        # -> Generator[Any, None, None] only for >py39
        # TODO: really consider how to handle multi-process/multi-node sharding
        # 1. Get the original stream
        stream = self._read_stream()

        # 2. Multi-process/multi-node sharding
        stream = self._shard_stream(stream)

        # 3. Skip resumed samples
        stream = self._skip_samples(stream)

        # 4. Filter
        if self.cfg.filter_fn is not None:
            stream = filter(self.cfg.filter_fn, stream)

        # 5. Transform
        if self.cfg.transform is not None:
            stream = map(self.cfg.transform, stream)

        # 6. Shuffle (streaming)
        if self.cfg.shuffle_buffer_size > 0:
            stream = self._shuffle_stream(stream)

        yield from stream

    def _shard_stream(self, stream: Iterable[Any]) -> Generator[Any, None, None]:
        """
        Shard the data stream based on worker and distributed rank, ensuring each sample is processed only once.
        """
        worker_info = get_worker_info()
        # Total parallelism = number of nodes * processes per node * workers per process
        total_parts = self.cfg.world_size
        part_id = self.cfg.rank

        if worker_info is not None:
            total_parts *= worker_info.num_workers
            part_id = part_id * worker_info.num_workers + worker_info.id

        for idx, sample in enumerate(stream):
            if idx % total_parts == part_id:
                yield sample

    def _skip_samples(self, stream: Iterable[Any]) -> Generator[Any, None, None]:
        """
        Skip samples before resume_from_sample.
        """
        if self.cfg.resume_from_sample <= 0:
            yield from stream
            return
        for idx, sample in enumerate(stream, start=1):
            if idx > self.cfg.resume_from_sample:
                yield sample

    def _shuffle_stream(self, stream: Iterable[Any]) -> Generator[Any, None, None]:
        """
        Use fixed-size buffer for streaming shuffle.
        """
        buf: List[Any] = []
        for sample in stream:
            buf.append(sample)
            if len(buf) >= self.cfg.shuffle_buffer_size:
                idx = self._rng.randrange(len(buf))
                yield buf.pop(idx)
        # Randomly output remaining samples
        self._rng.shuffle(buf)
        yield from buf

    def get_logger(self):
        return getLogger(self.__class__.__name__)
