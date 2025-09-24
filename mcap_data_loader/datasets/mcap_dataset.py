import os
from typing import Any, Generator, Iterable, Iterator, List, Optional, Dict, Union
from pydantic import field_validator
from functools import cache
import numpy as np
from mcap_data_loader.utils.mcap_utils import McapFlatBuffersReader
from mcap_data_loader.utils.basic import (
    get_items_by_ext,
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


class McapDatasetConfig(IterableDatasetConfig):
    """
    MCAP dataset configuration.
    """

    keys: List[str] = []
    topics: Optional[List[str]] = []
    attachments: Optional[List[str]] = []

    @field_validator("data_root")
    def validate_data_root(cls, v) -> str:
        if not isinstance(v, str):
            if len(v) == 1:
                v = v[0]
            else:
                raise ValueError(f"data_root {v} must be a string path to a MCAP file")
        if not v.endswith(".mcap"):
            raise ValueError(f"data_root {v} must be a `.mcap` file")
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


class McapFlatBuffersSampleDataset(IterableDatasetABC):
    """
    Iterable dataset for reading a MCAP file.
    """

    def __init__(self, config: McapDatasetConfig):
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

    def read_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Read MCAP file and return message stream.
        """
        yield from self.reader.iter_samples(
            keys=self.config.keys,
            topics=self.config.topics,
            attachments=self.config.attachments,
            reverse=self.config.rearrange.episode == RearrangeType.REVERSE,
        )

    def __del__(self):
        if self.reader:
            self.reader.close()

    def __len__(self) -> int:
        """Get the total number of messages in the MCAP file."""
        return len(self.reader) if self.reader else 0

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        return super().__iter__()


class McapFlatBuffersEpisodeDatasetConfig(McapDatasetConfig):
    """
    Episodic dataset configuration for reading MCAP files.
    """

    flatten: bool = False

    @field_validator("data_root")
    def validate_data_root(cls, v) -> List[str]:
        if isinstance(v, str):
            v = [v]
        for directory in v:
            if not os.path.isdir(directory):
                raise ValueError(
                    f"data_root {os.path.abspath(directory)} must be a directory containing MCAP files"
                )
        return v

    def model_post_init(self, context):
        assert not self.slices.sample, "not implemented yet"
        assert not self.slices.episode, "not implemented yet"
        assert isinstance(self.slices.dataset, dict), "dataset slices must be a dict"
        assert self.rearrange.sample == RearrangeType.NONE, (
            "sample rearrangement is not supported"
        )


class McapFlatBuffersEpisodeDataset(IterableDatasetABC):
    """
    Episodic dataset for reading MCAP files.
    """

    def __init__(self, config: McapFlatBuffersEpisodeDatasetConfig):
        super().__init__(config)
        self.config = config
        dataset_files = {}
        DataRearrangeConfig.rearrange(
            self.config.data_root, self.config.rearrange.dataset, self._rng
        )
        file_cnt = 0
        for root in self.config.data_root:
            files = get_items_by_ext(root, ".mcap")
            DataRearrangeConfig.rearrange(
                files, self.config.rearrange.episode, self._rng
            )
            indexes = self.config.slices.dataset_indexes.get(root, None)
            if indexes:
                # slice the files by indexes
                files = np.array(files)[indexes].tolist()
            dataset_files[root] = files
            file_cnt += len(files)
        if file_cnt == 0:
            raise ValueError(
                f"No MCAP files found in {self.config.data_root}, please check the path."
            )
        self._file_cnt = file_cnt
        self._dataset_files = dataset_files

    def read_stream(self) -> Generator[Iterable[dict[str, Any]], None, None]:
        """
        Read MCAP files and return episodic message stream.
        Each episode corresponds to one MCAP file.
        """
        cfg_dict = self.config.model_dump()
        cfg_dict.pop("data_root")
        for dataset, file_paths in self._dataset_files.items():
            for file_path in file_paths:
                full_path = os.path.join(dataset, file_path)
                sample_ds = McapFlatBuffersSampleDataset(
                    McapDatasetConfig(data_root=full_path, **cfg_dict)
                )
                sample_ds.load()
                if self.config.flatten:
                    yield from sample_ds
                else:
                    yield sample_ds

    @property
    def all_files(self) -> Dict[str, List[str]]:
        """Get all dataset files corresponding to each dataset root."""
        return self._dataset_files

    @cache
    def __len__(self) -> int:
        """Get the total number of episodes across all dataset roots."""
        if not self.config.flatten:
            return self._file_cnt
        else:
            return super().__len__()

    def __iter__(self) -> Iterator[McapFlatBuffersSampleDataset]:
        return super().__iter__()

    def __getitem__(
        self, index
    ) -> Union[Dict[str, np.ndarray], McapFlatBuffersSampleDataset]:
        return super().__getitem__(index)
