import numpy as np
import random
from pathlib import Path
from typing import List, Optional, Dict, Union, Any
from collections.abc import Sequence, Hashable
from typing_extensions import Self
from functools import cached_property, cache
from pydantic import (
    field_validator,
    ConfigDict,
    BaseModel,
    Field,
    FilePath,
    DirectoryPath,
)
from mcap_data_loader.serialization.flb import McapFlatBuffersReader
from mcap_data_loader.utils.basic import (
    get_items_by_ext,
    file_hash,
    DictDataStamped,
    SetMapping,
)
from mcap_data_loader.utils.stat import combine_dict_groups
from mcap_data_loader.datasets.dataset import (
    IterableDatasetABC,
    IterableDatasetConfig,
    DataSlicesConfig,
    DataRearrangeConfig,
    RearrangeType,
)


SampleType = Dict[str, np.ndarray]
SampleStamped = DictDataStamped[np.ndarray]
SampleUnion = Union[SampleType, SampleStamped]


class McapDatasetConfig(IterableDatasetConfig):
    """
    MCAP dataset configuration.
    """

    keys: SetMapping[str] = set()
    topics: Optional[SetMapping[str]] = set()
    attachments: Optional[SetMapping[str]] = set()
    with_timestamp: bool = True
    strict: bool = True
    media_configs: List = []


class McapFlatBuffersSampleDatasetConfig(McapDatasetConfig):
    """
    Sample dataset configuration for reading a MCAP file.
    """

    data_root: FilePath

    @field_validator("data_root")
    def validate_data_root(cls, v: Path) -> Path:
        if v.suffix != ".mcap":
            raise ValueError(f"data_root {v} must be an `.mcap` file")
        return v

    @field_validator("slices")
    def validate_slices(cls, v: DataSlicesConfig) -> DataSlicesConfig:
        if v != DataSlicesConfig():
            raise ValueError("slices are not supported now")
        return v

    @field_validator("rearrange")
    def validate_rearrange(cls, v: DataRearrangeConfig) -> DataRearrangeConfig:
        if (v.sample, v.dataset) != (RearrangeType.NONE, RearrangeType.NONE):
            raise ValueError("sample and dataset rearrangement are not supported")
        if v.episode not in {RearrangeType.NONE, RearrangeType.REVERSE}:
            raise ValueError("episode rearrangement must be NONE or REVERSE")
        return v


class McapFlatBuffersSampleDataset(IterableDatasetABC[SampleUnion]):
    """
    Iterable dataset for reading a MCAP file.
    """

    def __init__(self, config: McapFlatBuffersSampleDatasetConfig):
        self.config = config
        # delay the reader initialization until read_stream is called
        # to support multiprocessing data reading
        self.reader = None

    def read_stream(self):
        """
        Read MCAP file and return message stream.
        """
        self._ensure_reader()
        config = self.config
        samples_iter = self.reader.iter_samples(
            config.keys,
            config.topics,
            config.attachments,
            config.rearrange.episode is RearrangeType.REVERSE,
            config.strict,
            config.media_configs,
        )
        if self.config.with_timestamp:
            yield from samples_iter
        else:
            for sample in samples_iter:
                yield {key: value["data"] for key, value in sample.items()}

    def close(self):
        if self.reader is not None:
            return self.reader.close()

    def statistics(self):
        self._ensure_reader()
        return self.reader.topic_statistics

    def _ensure_reader(self):
        if self.reader is None:
            self.reader = McapFlatBuffersReader(open(self.config.data_root, "rb"))

    def __len__(self) -> int:
        """Get the total number of messages in the MCAP file."""
        return len(self.reader) if self.reader else 0

    def __lt__(self, other: Self) -> bool:
        return self.config.data_root < other.config.data_root

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config.data_root})"

    @property
    def stem(self) -> str:
        """Get the stem of the MCAP file."""
        return self.config.data_root.stem


class McapFlatBuffersEpisodeDatasetConfig(McapDatasetConfig):
    """
    Episodic dataset configuration for reading MCAP files.
    """

    data_root: DirectoryPath

    @field_validator("slices")
    def validate_slices(cls, v: DataSlicesConfig) -> DataSlicesConfig:
        if isinstance(v.dataset, dict):
            raise ValueError("slices.dataset can not be a dict")
        if v.sample or v.episode:
            raise ValueError("slices.sample and slices.episode must be empty")
        return v

    @field_validator("rearrange")
    def validate_rearrange(cls, v: DataRearrangeConfig) -> DataRearrangeConfig:
        if v.sample != RearrangeType.NONE:
            raise ValueError("sample rearrangement is not supported")
        return v


class McapFlatBuffersEpisodeDataset(IterableDatasetABC[McapFlatBuffersSampleDataset]):
    """
    Episodic dataset for reading MCAP files.
    """

    def __init__(self, config: McapFlatBuffersEpisodeDatasetConfig):
        self.config = config
        root = self.config.data_root
        files = get_items_by_ext(root, ".mcap")
        RearrangeType.rearrange(
            files,
            self.config.rearrange.dataset,
            random.Random(self.config.rearrange.seed),
        )
        indexes = self.config.slices.dataset_indexes.get(root, None)
        if indexes:
            # slice the files by indexes
            files = np.array(files)[indexes].tolist()
        if not files:
            raise ValueError(
                f"No MCAP files found in {self.config.data_root}, please check the path."
            )
        self._episode_files = files
        self._sample_ds_cfg = self.config.model_dump(
            exclude={"data_root", "media_configs"}
        )
        self._sample_ds_cfg["rearrange"] = config.rearrange.model_copy(
            update={"dataset": RearrangeType.NONE}
        )

    def read_stream(self):
        """
        Read MCAP files and return episodic message stream.
        Each episode corresponds to one MCAP file.
        """
        for file_path in self._episode_files:
            yield self._create_sample_dataset(file_path)

    def _create_sample_dataset(self, file_path: str) -> McapFlatBuffersSampleDataset:
        return McapFlatBuffersSampleDataset(
            McapFlatBuffersSampleDatasetConfig(
                data_root=file_path,
                # should not be a dict type
                media_configs=self.config.media_configs,
                **self._sample_ds_cfg,
            )
        )

    @cache
    def statistics(self):
        return combine_dict_groups(ds.statistics() for ds in self.read_stream())

    @property
    def all_files(self) -> List[str]:
        """Get all episode files."""
        return self._episode_files

    @cached_property
    def all_file_hashes(self) -> List[str]:
        """Get the hash values of all episode files."""
        return [file_hash(file_path) for file_path in self._episode_files]

    def __len__(self) -> int:
        """Get the total number of episodes across all dataset roots."""
        return len(self._episode_files)

    def __getitem__(self, index: int):
        return self._create_sample_dataset(self._episode_files[index])


def get_config_and_class_type(data_root: Path):
    """
    Get the appropriate dataset configuration and class type based on the data root.
    """
    if not data_root.exists():
        raise ValueError(f"data_root {data_root} does not exist.")
    if data_root.is_file():
        return McapFlatBuffersSampleDatasetConfig, McapFlatBuffersSampleDataset
    else:
        return McapFlatBuffersEpisodeDatasetConfig, McapFlatBuffersEpisodeDataset


def get_first_sample(
    dataset: Union[McapFlatBuffersSampleDataset, McapFlatBuffersEpisodeDataset],
    keys: Optional[List[str]] = None,
) -> SampleUnion:
    """
    Get the first sample from the dataset for the specified keys.
    """
    if not isinstance(dataset, McapFlatBuffersSampleDataset):
        # get the first episode dataset
        dataset = dataset[0]
    sample = next(iter(dataset.read_stream()))
    if keys is not None:
        sample = {key: sample[key] for key in keys}
    return sample


def to_episodic_sequence(dataset) -> Sequence[McapFlatBuffersSampleDataset]:
    if isinstance(dataset, McapFlatBuffersSampleDataset):
        return [dataset]
    return dataset


class McapMultiEpisodeDatasetsConfig(BaseModel, frozen=True):
    """
    Multi-episodic dataset configuration for reading MCAP files from multiple roots.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    common: Dict[str, Any] = {}
    """Common configuration for all dataset roots."""
    configs: Dict[Hashable, Dict[str, Any]] = Field(min_length=1)
    """Dataset configurations. The key is the unique name of the dataset, and the value is the dataset config dict."""


class McapMultiEpisodeDatasets(IterableDatasetABC[McapFlatBuffersEpisodeDataset]):
    """
    Multi-episodic dataset for reading MCAP files from multiple roots.
    """

    def __init__(self, config: McapMultiEpisodeDatasetsConfig):
        self.config = config
        self._episode_datasets: List[McapFlatBuffersEpisodeDataset] = []
        self._init_episode_datasets()

    def _init_episode_datasets(self):
        """
        Initialize episode datasets from multiple roots.
        """
        for name, cfg in self.config.configs.items():
            self.get_logger().debug(f"Initializing dataset '{name}' with config: {cfg}")
            data_root = Path(cfg["data_root"])
            merge_config = self.config.common.copy()
            merge_config.update(cfg)
            config_cls, dataset_cls = get_config_and_class_type(data_root)
            episode_dataset = dataset_cls(config_cls(**merge_config))
            episode_dataset = to_episodic_sequence(episode_dataset)
            self._episode_datasets.append(episode_dataset)

    def read_stream(self):
        """
        Read MCAP files from multiple roots and return episodic message stream.
        """
        return iter(self._episode_datasets)

    @cache
    def statistics(self):
        return combine_dict_groups(ds.statistics() for ds in self.read_stream())

    def __len__(self) -> int:
        """Get the total number of episodes across all dataset roots."""
        return len(self._episode_datasets)
