import os
from typing import Any, Generator, Iterable, Iterator, List, Optional, Dict
from pydantic import field_validator
from functools import cache
import numpy as np
from more_itertools import peekable, nth
from mcap_data_loader.utils.mcap_utils import McapFlatbufferReader
from mcap_data_loader.utils.basic import (
    get_items_by_ext,
    zip,
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
    cache_items: bool = True
    cache_iters: bool = False

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
        assert not self.cache_iters, "iters now are not cached"
        assert self.rearrange.sample == RearrangeType.NONE, (
            "sample rearrangement is not supported"
        )
        assert self.rearrange.episode in {RearrangeType.NONE, RearrangeType.REVERSE}, (
            "episode rearrangement must be NONE or REVERSE"
        )
        assert self.rearrange.dataset == RearrangeType.NONE, (
            "dataset rearrangement is not supported"
        )


class McapFlatbufferSampleDataset(IterableDatasetABC):
    """
    Iterable dataset for reading a MCAP file.
    """

    cfg: McapDatasetConfig

    def load(self):
        self._init_reader()
        if self.cfg.cache_items:
            self._indexed_stream = peekable(self._flatten_iter())

    def _flatten_iter(self):
        """Flatten"""
        return self

    def _init_reader(self):
        """
        Initialize the MCAP reader.
        This is called in the constructor to set up the reader.
        """
        self.reader = McapFlatbufferReader(open(self.cfg.data_root, "rb"))

    def _read_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Read MCAP file and return message stream.
        """
        return self._iter_a_file_samples(self.reader)

    def _iter_a_file_samples(
        self, reader: McapFlatbufferReader
    ) -> Generator[Dict[str, Any], None, None]:
        yield from reader.iter_samples(
            keys=self.cfg.keys,
            topics=self.cfg.topics,
            attachments=self.cfg.attachments,
            reverse=self.cfg.rearrange.episode == RearrangeType.REVERSE,
        )

    def __del__(self):
        if hasattr(self, "reader"):
            if self.reader:
                self.reader.file_io.close()

    def __len__(self) -> int:
        """Get the total number of messages in the MCAP file."""
        return len(self.reader)

    def __getitem__(self, index: int):
        """
        Get a specific sample by index.
        This is not efficient for large datasets, use with caution.
        """
        # TODO: should support 2-dim indexing, e.g.
        # dataset[episode_index][sample_index] or
        # dataset[episode_index, sample_index]?
        # This may be configurable in the future.
        if index < 0:
            index += len(self)
        if self.cfg.cache_items:
            return self._indexed_stream[index]
        else:
            return nth(self._flatten_iter(), index)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        return super().__iter__()


class McapFlatbufferEpisodeDatasetConfig(McapDatasetConfig):
    """
    Episodic dataset configuration for reading MCAP files.
    """

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
        assert not self.cache_iters, "iters now are not cached"
        assert self.rearrange.sample == RearrangeType.NONE, (
            "sample rearrangement is not supported"
        )


class McapFlatbufferEpisodeDataset(McapFlatbufferSampleDataset):
    """
    Episodic dataset for reading MCAP files.
    """

    cfg: McapFlatbufferEpisodeDatasetConfig

    def __init__(self, config):
        super().__init__(config)
        self.reader: Dict[str, McapFlatbufferReader] = {}
        dataset_files = {}
        DataRearrangeConfig.rearrange(
            self.cfg.data_root, self.cfg.rearrange.dataset, self._rng
        )
        for root in self.cfg.data_root:
            files = get_items_by_ext(root, ".mcap")
            DataRearrangeConfig.rearrange(files, self.cfg.rearrange.episode, self._rng)
            indexes = self.cfg.slices.dataset_indexes.get(root, None)
            if indexes:
                # slice the files by indexes
                files = np.array(files)[indexes].tolist()
            dataset_files[root] = files
        self._dataset_files = dataset_files

    def _flatten_iter(self):
        for episode in self:
            for sample in episode:
                yield sample

    def _init_reader(self):
        for dataset, file_paths in self._dataset_files.items():
            for file_path in file_paths:
                full_path = os.path.join(dataset, file_path)
                assert full_path not in self.reader, f"Duplicate file path: {full_path}"
                self.reader[full_path] = McapFlatbufferReader(open(full_path, "rb"))

    def _read_stream(self) -> Generator[Iterable[dict[str, Any]], None, None]:
        """
        Read MCAP files and return episodic message stream.
        Each episode corresponds to one MCAP file.
        """
        for file_path, reader in self.reader.items():
            self._current_file = file_path
            yield self._iter_a_file_samples(reader)

    @property
    def current_file(self) -> str:
        return self._current_file

    @property
    def all_files(self) -> Dict[str, List[str]]:
        return self._dataset_files

    def __del__(self):
        for reader in self.reader.values():
            reader.file_io.close()

    @cache
    def __len__(self) -> int:
        """Get the total number of messages in all MCAP files."""
        total_count = 0
        for reader in self.reader.values():
            total_count += len(reader)
        return total_count

    def __iter__(self) -> Iterator[Iterator[Dict[str, np.ndarray]]]:
        return super().__iter__()

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        return super().__getitem__(index)


if __name__ == "__main__":
    from pprint import pprint
    import time
    from more_itertools import batched
    import logging
    from mcap_data_loader.datasets.dataset import DataSlicesConfig

    logging.basicConfig(level=logging.INFO)

    root_dir = "data/arm1-001"
    # data_root = "0.mcap"
    data_root = root_dir
    keys = [
        "/left/follow/arm/joint_state/position",
        "/left/follow/eef/joint_state/position",
        "/left/lead/arm/joint_state/position",
        "/left/lead/eef/joint_state/position",
        "/env_camera/env/color/image_raw",
    ]
    # keys = (
    #     [
    #         # "/follow/arm/joint_state/position",
    #         # "/follow/eef/joint_state/position",
    #     ]
    #     + [
    #         "/env_camera/color/image_raw",
    #         # "/follow_camera/color/image_raw",
    #         # discoverse camera keys
    #         # "/cam_0/color/image_raw",
    #         # "/cam_1/color/image_raw",
    #         "log_stamps",
    #     ]
    # )

    # dataset = McapFlatbufferDataset(
    #     McapFlatbufferDatasetConfig(
    #         data_root=data_root,
    #         keys=keys,
    #     )
    # )
    # start = time.perf_counter()
    # for sample in dataset:
    #     print(time.perf_counter() - start)
    #     # pprint(sample)
    #     start = time.perf_counter()
    #     # break  # Only print the first sample

    dataset = McapFlatbufferEpisodeDataset(
        McapFlatbufferEpisodeDatasetConfig(
            data_root=data_root,
            keys=keys,
            slices=DataSlicesConfig(dataset={root_dir: (0, 1)}),
            rearrange=DataRearrangeConfig(
                episode="sort",
            ),
            cache_items=True,
        )
    )
    dataset.load()
    print(dataset.all_files)
    print(f"Dataset length: {len(dataset)}")
    pprint(dataset[0].keys())
    for v1, v2 in zip(dataset[0].values(), dataset[0].values()):
        assert np.array_equal(v1, v2), f"{v1=} != {v2=}"
    for v1, v2 in zip(dataset[0].values(), dataset[1].values()):
        if not np.array_equal(v1, v2):
            print("OK: Samples are not equal")
            break
    else:
        raise ValueError("Samples are equal")

    for file_path, reader in dataset.reader.items():
        print(f"File: {file_path}, Messages: {len(reader)}")
    start = time.perf_counter()
    batch_size = 10
    steps = 1
    for episode in dataset:
        next(episode)  # Skip the first sample
        start = time.perf_counter()
        for step, batch in enumerate(batched(episode, batch_size, strict=True)):
            print(f"{step=}", batch[0].keys())
            if step + 1 >= steps:
                break
        else:
            print(f"Processed {len(episode)} samples in episode {dataset.current_file}")
        total_time = time.perf_counter() - start
        avg_time = total_time / batch_size
        print(f"Average time per sample: {avg_time:.5f} seconds")
        print(f"Total time taken for {batch_size=}: {total_time:.5f} seconds")
        break  # Only process the first episode
