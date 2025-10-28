import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
from typing_extensions import Self
from functools import cached_property
from pydantic import field_validator
from mcap_data_loader.serialization.flb import McapFlatBuffersReader
from mcap_data_loader.utils.basic import (
    get_items_by_ext,
    file_hash,
    DictDataStamped,
    # zip,
    # DictableSlicesType,
    # DictableIndexesType,
)
from mcap_data_loader.datasets.dataset import (
    IterableDatasetABC,
    IterableDatasetConfig,
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

    def model_post_init(self, context):
        assert not self.slices.sample, "not implemented yet"
        assert not self.slices.episode, "not implemented yet"
        assert isinstance(self.slices.dataset, dict), "dataset slices must be a dict"
        assert self.rearrange.sample == RearrangeType.NONE, (
            "sample rearrangement is not supported"
        )
        assert self.rearrange.episode in {RearrangeType.NONE, RearrangeType.REVERSE}, (
            "episode rearrangement must be NONE or REVERSE"
        )
        assert self.rearrange.dataset == RearrangeType.NONE, (
            "dataset rearrangement is not supported"
        )


class McapFlatBuffersSampleDataset(IterableDatasetABC[SampleUnion]):
    """
    Iterable dataset for reading a MCAP file.
    """

    def __init__(self, config: McapFlatBuffersSampleDatasetConfig):
        super().__init__(config)
        self.config = config
        self.reader = None

    def load(self):
        self._init_reader()
        return super().load()

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
            self.config.rearrange.episode == RearrangeType.REVERSE,
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

    def model_post_init(self, context):
        assert not self.slices.sample, "not implemented yet"
        assert not self.slices.episode, "not implemented yet"
        assert isinstance(self.slices.dataset, dict), "dataset slices must be a dict"
        assert self.rearrange.sample == RearrangeType.NONE, (
            "sample rearrangement is not supported"
        )


class McapFlatBuffersEpisodeDataset(IterableDatasetABC[McapFlatBuffersSampleDataset]):
    """
    Episodic dataset for reading MCAP files.
    """

    def __init__(self, config: McapFlatBuffersEpisodeDatasetConfig):
        super().__init__(config)
        self.config = config
        root = self.config.data_root
        files = get_items_by_ext(root, ".mcap")
        DataRearrangeConfig.rearrange(files, self.config.rearrange.dataset, self._rng)
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
            sample_ds = self._create_sample_dataset(file_path)
            yield sample_ds

    def _create_sample_dataset(self, file_path: str) -> McapFlatBuffersSampleDataset:
        sample_ds = McapFlatBuffersSampleDataset(
            McapDatasetConfig(
                data_root=file_path,
                media_configs=self.config.media_configs,
                **self._sample_ds_cfg,
            )
        )
        sample_ds.load()
        return sample_ds

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
