import numpy as np
import random
from pathlib import Path
from typing import List, Optional, Dict, Union, Sequence, Any
from typing_extensions import Self
from functools import cached_property
from pydantic import field_validator, ConfigDict, BaseModel, Field
from mcap_data_loader.serialization.flb import McapFlatBuffersReader
from mcap_data_loader.utils.basic import get_items_by_ext, file_hash, DictDataStamped
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

    model_config = ConfigDict(validate_assignment=True)

    keys: List[str] = []
    topics: Optional[List[str]] = []
    attachments: Optional[List[str]] = []
    with_timestamp: bool = True
    strict: bool = True
    media_configs: List = []


class McapFlatBuffersSampleDatasetConfig(McapDatasetConfig):
    """
    Sample dataset configuration for reading a MCAP file.
    """

    @field_validator("data_root")
    def validate_data_root(cls, v) -> Path:
        if not isinstance(v, Path):
            if len(v) == 1:
                v = v[0]
            else:
                raise ValueError(f"data_root {v} must be a single path to a MCAP file")
        if not v.is_file() or v.suffix != ".mcap":
            raise ValueError(f"data_root {v} must be an existing `.mcap` file")
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
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        """
        Initialize the MCAP reader.
        This is called in the constructor to set up the reader.
        """
        self.reader = McapFlatBuffersReader(open(self.config.data_root, "rb"))

    def read_stream(self):
        """
        Read MCAP file and return message stream.
        """
        samples_iter = self.reader.iter_samples(
            self.config.keys,
            self.config.topics,
            self.config.attachments,
            self.config.rearrange.episode is RearrangeType.REVERSE,
            self.config.strict,
            self.config.media_configs,
        )
        if self.config.with_timestamp:
            yield from samples_iter
        else:
            for sample in samples_iter:
                yield {key: value["data"] for key, value in sample.items()}

    def __del__(self):
        if self.reader:
            self.reader.close()

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

    @field_validator("data_root")
    def validate_data_root(cls, v: Path) -> List[Path]:
        if not v.is_dir():
            raise ValueError(
                f"data_root {v.absolute()} must be a directory containing MCAP files"
            )
        return v

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

    def read_stream(self):
        """
        Read MCAP files and return episodic message stream.
        Each episode corresponds to one MCAP file.
        """
        for file_path in self._episode_files:
            yield self._create_sample_dataset(file_path)

    def _create_sample_dataset(self, file_path: str) -> McapFlatBuffersSampleDataset:
        return McapFlatBuffersSampleDataset(
            McapDatasetConfig(
                data_root=file_path,
                media_configs=self.config.media_configs,
                **self._sample_ds_cfg,
            )
        )

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


class McapMultiEpisodeDatasetsConfig(BaseModel):
    """
    Multi-episodic dataset configuration for reading MCAP files from multiple roots.
    """

    common: Dict[str, Any] = {}
    """Common configuration for all dataset roots."""
    roots: Dict[str, Union[Dict[str, Any], List[str]]] = Field(min_length=1)
    """Dataset root specific configurations. The key is the root path, and the value is either a dict of configurations or a list of keys to read."""

    @field_validator("roots")
    def validate_roots(cls, v: Dict[str, Any]) -> dict:
        for root in list(v.keys()):
            cfg = v[root]
            splited = root.split("//", 1)
            if len(splited) == 2:
                v.pop(root)
                root = splited[1]
                base_cfg = v.pop(root, {})
            else:
                base_cfg = {}
            if isinstance(cfg, list):
                if len(cfg) == 0:
                    raise ValueError(f"root[{root}] keys must not be an empty list")
                dict_cfg = {"keys": cfg}
            else:
                dict_cfg = cfg
            # merge keys and overwrite the config
            keys = base_cfg.get("keys", []) + dict_cfg.get("keys", [])
            base_cfg.update(dict_cfg)
            v[root] = base_cfg
            if keys:
                v[root]["keys"] = keys
            # if not v[root]:
            #     raise ValueError(f"root[{root}] configuration must not be empty")
        return v


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
        for root_str, cfg in self.config.roots.items():
            data_root = Path(root_str)
            merge_config = self.config.common.copy()
            merge_config.update(cfg)
            config_cls, dataset_cls = get_config_and_class_type(data_root)
            dataset_cfg = config_cls(
                data_root=data_root,
                **merge_config,
            )
            episode_dataset = dataset_cls(dataset_cfg)
            episode_dataset = to_episodic_sequence(episode_dataset)
            self._episode_datasets.append(episode_dataset)

    def read_stream(self):
        """
        Read MCAP files from multiple roots and return episodic message stream.
        """
        yield from self._episode_datasets

    def __len__(self) -> int:
        """Get the total number of episodes across all dataset roots."""
        return len(self._episode_datasets)
