from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch import Tensor, asarray
import torch
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Union, Dict, Iterable
from functools import cached_property
from collections.abc import Mapping
from collections import deque
import random
from queue import Queue, Empty, Full
from threading import Thread, Event
from datetime import datetime
from pathlib import Path
import json
import os
from mcap_data_loader.datasets.mcap_dataset import (
    McapMultiEpisodeDatasets,
    McapMultiEpisodeDatasetsConfig,
    SampleStamped,
)
from mcap_data_loader.utils.basic import force_set_attr
from mcap_data_loader.utils.stat import concatenate_statistics, Statistics
from mcap_data_loader.pipelines import HorizonConfig
from mcap_data_loader.utils.av_coder import DecodeConfig
from ast import literal_eval


class McapLeRobotDatasetConfig(BaseModel, frozen=True):
    """Configuration for McapLeRobotDataset."""

    model_config = ConfigDict(validate_assignment=True)

    data_root: Union[str, Dict[str, List[str]], List[str]]
    """The root directories of the dataset."""
    states: List[str] = []
    """The list of state keys."""
    images: List[str] = []
    """The list of image keys."""
    actions: List[str]
    """The list of action keys."""
    horizon: HorizonConfig = {}
    """The horizon configuration."""
    shuffle_episodes: bool = False
    """Whether to shuffle the merged episode order for each new pass."""
    shuffle_seed: int = 0
    """Base random seed for episode-level shuffling."""
    prefetch_items: int = 0
    """Number of items to prefetch on a background thread. Set to 0 to disable."""

    @field_validator("horizon", mode="before")
    def validate_horizon(cls, v):
        # use a validator to change the default value
        if isinstance(v, Mapping):
            return {"fill_with_last": True} | v
        return v

    @force_set_attr
    def model_post_init(self, context):
        if isinstance(self.data_root, str):
            self.data_root = {self.data_root: self.states + self.images + self.actions}
        elif isinstance(self.data_root, list):
            self.data_root = {
                root: self.states + self.images + self.actions
                for root in self.data_root
            }


class McapLeRobotDatasetMeta(BaseModel, frozen=True):
    """Metadata for McapLeRobotDataset."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    features: Dict[str, Dict[str, Union[tuple, str, Dict[str, List[str]]]]]
    """The dictionary of features, where the key is the feature name and the value is a dictionary containing the shape and dtype of the feature."""
    stats: Dict[str, Statistics]
    """The dictionary of statistics, where the key is the feature name and the value is a dictionary containing the statistics of the feature."""
    camera_keys: List[str] = []
    """The list of camera keys."""

    @field_validator("stats", mode="after")
    def validate_stats(cls, v: dict):
        for stat in v.values():
            stat["count"] = stat["n"]
        return v


ItemType = Dict[str, Tensor]

STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEY_PREFIX = "observation.images"
_QUEUE_SENTINEL = object()


def _read_proc_status_value_mb(field_name: str) -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith(field_name):
                    value_kb = int(line.split()[1])
                    return value_kb / 1024.0
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None
    return None


def _get_memory_snapshot() -> dict[str, float | int | str | None]:
    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "rss_mb": _read_proc_status_value_mb("VmRSS:"),
        "hwm_mb": _read_proc_status_value_mb("VmHWM:"),
    }
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["cuda_allocated_mb"] = (
                torch.cuda.memory_allocated() / 1024.0 / 1024.0
            )
            snapshot["cuda_reserved_mb"] = (
                torch.cuda.memory_reserved() / 1024.0 / 1024.0
            )
            snapshot["cuda_max_allocated_mb"] = (
                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            )
            snapshot["cuda_max_reserved_mb"] = (
                torch.cuda.max_memory_reserved() / 1024.0 / 1024.0
            )
    except Exception:
        pass
    return snapshot


def _start_memory_logger(
    log_dir: Path, interval_s: int = 30
) -> tuple[Path, Event, Thread]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"memory_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    stop_event = Event()

    def writer():
        with open(log_path, "a", encoding="utf-8") as f:
            while not stop_event.is_set():
                f.write(json.dumps(_get_memory_snapshot(), ensure_ascii=True) + "\n")
                f.flush()
                stop_event.wait(interval_s)
            f.write(json.dumps(_get_memory_snapshot(), ensure_ascii=True) + "\n")
            f.flush()

    thread = Thread(target=writer, name="mcap-memory-logger", daemon=True)
    thread.start()
    return log_path, stop_event, thread


class McapLeRobotDataset(IterableDataset):
    def __init__(self, config: McapLeRobotDatasetConfig):
        self.config = config
        self._state_keys = tuple(config.states)
        self._image_keys = tuple(config.images)
        self._action_keys = tuple(config.actions)
        self._horizon = config.horizon
        self._camera_mappings = {
            IMAGE_KEY_PREFIX + "." + img_key.removeprefix("/").split("/")[0]: img_key
            for img_key in config.images
        }
        self._timestamp_key = (self._state_keys + self._image_keys + self._action_keys)[
            0
        ]

        self._datasets = self._make_datasets()
        self._ds_iter = None
        self._epoch = 0
        first_item = next(self._iter_items(self._datasets, shuffle=False))
        # print(f"First item keys: {first_item.keys()}")
        self._add_items = {
            "action_is_pad": asarray([False] * first_item[ACTION_KEY].shape[0]),
        }
        # NOTE: there is a bug in lerobot act model that we must use a dummy state
        if not config.states:
            self._add_items[STATE_KEY] = asarray([])
        # NOTE: do not update the `first_item` here, otherwise the `action_is_pad` will
        # be treated as an action feaute
        # first_item.update(self._add_items)
        # NOTE: the `timestamp` in lerobot dataset is float32, but here
        # it is int64
        features = {}
        for key, value in first_item.items():
            if len(value.shape) == 3:
                dtype = "image"
                names = ["channel", "height", "width"]
                shape = tuple(value.shape)
            else:
                dtype = str(value.dtype).split(".")[-1]
                names = {"motors": []}
                if len(value.shape) > 1:
                    shape = (value.shape[-1],)
                else:
                    shape = value.shape
            features[key] = {"shape": shape, "dtype": dtype, "names": names}
        stats = {}
        data_stats = self._datasets.statistics()
        for concat_key, keys in zip(
            (STATE_KEY, ACTION_KEY), (config.states, config.actions)
        ):
            if not keys:
                continue
            stat = concatenate_statistics([data_stats[key] for key in keys])
            stat["count"] = stat["n"]
            stats[concat_key] = stat
        # add empty img stats to support modify stats outside
        for img_key in self._camera_mappings:
            # NOTE: assume the data length is the same
            stats[img_key] = Statistics.empty((3, 1, 1)) | {"count": stat["n"]}
        self._meta = McapLeRobotDatasetMeta(
            features=features, stats=stats, camera_keys=self._camera_mappings.keys()
        )

    def _make_datasets(self) -> McapMultiEpisodeDatasets:
        return McapMultiEpisodeDatasets(
            McapMultiEpisodeDatasetsConfig(
                common={
                    # "with_file": True,
                    "extra_keys": True,
                    "media_configs": [
                        DecodeConfig(
                            frame_format="rgb24",
                            dimension_order="NCHW",
                            backend="torchcodec",
                        ),
                    ],
                },
                configs={
                    data_root: {"data_root": data_root, "keys": keys}
                    for data_root, keys in self.config.data_root.items()
                },
            )
        )

    def _merge_episode_samples(self, sample_datasets: list) -> Iterable[SampleStamped]:
        streams = [dataset.read_stream() for dataset in sample_datasets]
        for samples in zip(*streams):
            merged_sample = {}
            for sample in samples:
                merged_sample.update(sample)
            yield merged_sample

    @staticmethod
    def _sample_image_tensor(sample: SampleStamped, key: str) -> Tensor:
        value = sample[key]["data"]
        assert isinstance(value, torch.Tensor), f"{key} must be a torch.Tensor"
        assert value.dtype == torch.uint8, f"{key} must be uint8, but got {value.dtype}"
        assert value.ndim == 3, f"{key} must be CHW"
        return value

    def _iter_episode_horizon_items(
        self, sample_datasets: list
    ) -> Iterable[tuple[tuple[SampleStamped, ...], tuple[SampleStamped, ...]]]:
        sample_iter = self._merge_episode_samples(sample_datasets)
        iterator = iter(sample_iter)
        try:
            first_sample = next(iterator)
        except StopIteration:
            return

        past_num = self._horizon.past_num
        gap = self._horizon.gap
        future_num = self._horizon.future_num
        window_size = past_num + gap + future_num + 1
        window = deque([first_sample] * past_num, maxlen=window_size)
        window.append(first_sample)

        initial_future_real = 0
        last_real = first_sample
        for _ in range(gap + future_num):
            try:
                next_sample = next(iterator)
                initial_future_real += 1
                last_real = next_sample
            except StopIteration:
                if self._horizon.fill_with_last:
                    next_sample = last_real
                else:
                    next_sample = self._horizon.fillvalue
            window.append(next_sample)

        def build_horizon_item():
            window_tuple = tuple(window)
            return (
                window_tuple[: past_num + 1],
                window_tuple[past_num + gap :],
            )

        current_index = 0
        if current_index % self._horizon.step == 0:
            yield build_horizon_item()

        for next_sample in iterator:
            last_real = next_sample
            window.append(next_sample)
            current_index += 1
            if current_index % self._horizon.step == 0:
                yield build_horizon_item()

        tail_fill = (
            last_real if self._horizon.fill_with_last else self._horizon.fillvalue
        )
        for _ in range(initial_future_real):
            window.append(tail_fill)
            current_index += 1
            if current_index % self._horizon.step == 0:
                yield build_horizon_item()

    @staticmethod
    def _stack_sample_keys(
        samples: tuple[SampleStamped, ...], keys: tuple[str, ...]
    ) -> Tensor:
        if len(keys) == 1:
            stacked = np.stack([sample[keys[0]]["data"] for sample in samples])
            return torch.from_numpy(stacked).to(dtype=torch.float32)

        stacked = np.stack(
            [
                np.concatenate([sample[key]["data"] for key in keys], axis=-1)
                for sample in samples
            ]
        )
        return torch.from_numpy(stacked).to(dtype=torch.float32)

    @staticmethod
    def _sample_key_tensor(sample: SampleStamped, key: str) -> Tensor:
        return McapLeRobotDataset._sample_image_tensor(sample, key)

    @staticmethod
    def _sample_keys_tensor(sample: SampleStamped, keys: tuple[str, ...]) -> Tensor:
        if len(keys) == 1:
            value = sample[keys[0]]["data"]
            return torch.from_numpy(value).to(dtype=torch.float32)
        merged = np.concatenate([sample[key]["data"] for key in keys], axis=-1)
        return torch.from_numpy(merged).to(dtype=torch.float32)

    def _stack_horizon_item(
        self, horizon_item: tuple[tuple[SampleStamped, ...], tuple[SampleStamped, ...]]
    ) -> ItemType:
        past, future = horizon_item
        current_sample = past[-1]
        item = {
            ACTION_KEY: self._stack_sample_keys(future, self._action_keys),
            "timestamp": asarray(current_sample[self._timestamp_key]["t"]),
        }
        if self._state_keys:
            item[STATE_KEY] = self._sample_keys_tensor(current_sample, self._state_keys)
        for camera_key, source_key in self._camera_mappings.items():
            item[camera_key] = self._sample_key_tensor(current_sample, source_key)
        return item

    def _iter_items(
        self, datasets: McapMultiEpisodeDatasets, shuffle: bool | None = None
    ):
        # Build and consume one merged episode stream at a time so image buffers
        # from previous episodes can be released promptly.
        episode_num = len(datasets._episode_datasets[0])
        episode_indices = list(range(episode_num))
        if shuffle is None:
            shuffle = self.config.shuffle_episodes
        if shuffle and episode_num > 1:
            random.Random(self.config.shuffle_seed + self._epoch).shuffle(
                episode_indices
            )
        for episode_index in episode_indices:
            sample_datasets = [
                episode_dataset[episode_index]
                for episode_dataset in datasets._episode_datasets
            ]
            for horizon_item in self._iter_episode_horizon_items(sample_datasets):
                yield self._stack_horizon_item(horizon_item)

    def _iter_prefetched(self, item_iter):
        queue = Queue(maxsize=max(self.config.prefetch_items, 1))
        stop_event = Event()

        def producer():
            try:
                for item in item_iter:
                    if stop_event.is_set():
                        break
                    while not stop_event.is_set():
                        try:
                            queue.put(item, timeout=0.1)
                            break
                        except Full:
                            continue
                if not stop_event.is_set():
                    queue.put(_QUEUE_SENTINEL)
            except BaseException as exc:  # pragma: no cover - forwarded to consumer
                if not stop_event.is_set():
                    queue.put(exc)

        thread = Thread(target=producer, name="mcap-prefetch", daemon=True)
        thread.start()
        try:
            while True:
                try:
                    value = queue.get(timeout=0.1)
                except Empty:
                    if stop_event.is_set() and not thread.is_alive():
                        break
                    continue
                if value is _QUEUE_SENTINEL:
                    break
                if isinstance(value, BaseException):
                    raise value
                yield value
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

    def _build_item_iter(self):
        item_iter = self._iter_items(self._datasets)
        return item_iter

    def collate_fn(self, batch: list[ItemType]) -> ItemType:
        collated = default_collate(batch)
        for camera_key in self._camera_mappings:
            collated[camera_key] = collated[camera_key].float().mul_(1.0 / 255.0)
        return collated

    def __iter__(self):
        item_iter = self._build_item_iter()
        self._epoch += 1
        for item in item_iter:
            item.update(self._add_items)
            yield item

    def __getitem__(self, index):
        if index == 0 or self._ds_iter is None:
            self._ds_iter = self._build_item_iter()
            self._epoch += 1
        item = next(self._ds_iter)
        item.update(self._add_items)
        return item

    @property
    def meta(self) -> McapLeRobotDatasetMeta:
        return self._meta

    @cached_property
    def num_frames(self) -> int:
        # NOTE: now the dataset iter times must be equal to the total frames
        # NOTE: now all the topic number must be same, so we just take the first one
        return next(iter(self._meta.stats.values()))["n"]

    @cached_property
    def num_episodes(self) -> int:
        # NOTE: the episode number must all be same so we just take the first one
        return len(self._datasets._episode_datasets[0])


def make_dataset(
    cfg, config_root: str = "configs", config_name: str = "config.yaml"
) -> McapLeRobotDataset:

    from pathlib import Path
    from mcap_data_loader.utils.hydra_utils import hydra_instance_from_config_path
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import DirectoryPath

    class ConfigSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env", env_file_encoding="utf-8", extra="ignore"
        )

        cfg_root: DirectoryPath = Path(config_root)
        cfg_name: str = config_name

        @property
        def cfg_path(self) -> Path:
            return self.cfg_root / Path(self.cfg_name).with_suffix(".yaml")

    settings = ConfigSettings()
    # the config file has the hightest priority
    print(f"Loading config from {settings.cfg_path}")
    dict_config = hydra_instance_from_config_path(settings.cfg_path)
    # cfg.dataset.episodes: list[int] | None
    try:
        data_dirs = literal_eval(cfg.dataset.repo_id)
    except SyntaxError:
        data_dirs = [cfg.dataset.repo_id]
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    # print(f"Data directories: {data_dirs}")
    data_paths = [str(Path(cfg.dataset.root) / data_dir) for data_dir in data_dirs]
    # print(f"Loading dataset from {data_path}")
    action_delta_indices = cfg.policy.action_delta_indices
    action_num = len(action_delta_indices)
    if action_num != action_delta_indices[-1] + 1:
        raise NotImplementedError(
            f"The action_num should be equal to the max index in action_delta_indices + 1, but got {action_num} and {action_delta_indices}"
        )
    # print(data_paths)
    base_dict = {"data_root": data_paths}
    if "mcap" in dict_config:
        # print("use mcap field")
        dict_config = dict_config["mcap"]
    base_dict.update(dict_config)
    return McapLeRobotDataset(
        McapLeRobotDatasetConfig(
            **base_dict,
            horizon=HorizonConfig(fill_with_last=True, future_num=action_num - 1),
        )
    )


def _process_config_path(path):
    from pathlib import Path

    path = Path(path)
    return str(path.parent), path.stem


def _has_cli_override(argv: list[str], prefix: str) -> bool:
    return any(arg == prefix or arg.startswith(f"{prefix}=") for arg in argv)


def _prefer_pyav_when_torchcodec_unavailable(argv: list[str]) -> None:
    import logging

    if _has_cli_override(argv, "--dataset.video_backend"):
        return

    try:
        import torchcodec  # noqa: F401
    except Exception as exc:
        argv.append("--dataset.video_backend=pyav")
        summary = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        logging.warning(
            "Falling back to dataset.video_backend=pyav because torchcodec is unavailable (%s). "
            "Set --dataset.video_backend=torchcodec to force it after fixing the local FFmpeg / TorchCodec runtime.",
            summary,
        )


def train():
    from lerobot.scripts import lerobot_train
    from mcap_data_loader.datasets.mcap_lerobot import make_dataset
    from mcap_data_loader.utils.cli import extract_and_remove_args, extend_args
    from functools import partial
    from mcap_data_loader.scripts.run_with_yaml import parse_args, get_args_list
    import logging
    import sys
    import torch

    argv_set = set(sys.argv)
    if ("--ori" not in argv_set) or ({"-c", "--config"} & argv_set):
        args = parse_args(["mcap"], False, False)
        if args is not None:
            _, extracted_dict = extract_and_remove_args(["-c", "--config", "--ori"])
            # print(f"Extracted args: {extracted_dict}")
            config_root, config_name = _process_config_path(args.config)

            if "--ori" not in extracted_dict:
                lerobot_train.make_dataset = partial(
                    make_dataset, config_root=config_root, config_name=config_name
                )

            # the cli args in lerobot_train will override the config file args
            extend_args(sys.argv, get_args_list(args))
    _prefer_pyav_when_torchcodec_unavailable(sys.argv)
    original_dataloader = torch.utils.data.DataLoader

    class McapAwareDataLoader(original_dataloader):
        @classmethod
        def __class_getitem__(cls, item):
            return original_dataloader[item]

        def __init__(self, dataset, *args, **kwargs):
            if (
                isinstance(dataset, McapLeRobotDataset)
                and kwargs.get("collate_fn") is None
            ):
                kwargs["collate_fn"] = dataset.collate_fn
            super().__init__(dataset, *args, **kwargs)

    torch.utils.data.DataLoader = McapAwareDataLoader
    # print(sys.argv), exit(0)
    log_path, stop_event, thread = _start_memory_logger(Path("outputs") / "memory_logs")
    logging.warning("Periodic memory logs will be written to %s", log_path)
    try:
        return lerobot_train.main()
    finally:
        torch.utils.data.DataLoader = original_dataloader
        stop_event.set()
        thread.join(timeout=1.0)


def run_with_yaml():
    from mcap_data_loader.scripts.run_with_yaml import parse_args, main_func
    import os

    args = parse_args(exclude=["mcap"])
    cfg_root, cfg_name = _process_config_path(args.config)
    if "cfg_root" not in os.environ:
        os.environ["cfg_root"] = cfg_root
    if "cfg_name" not in os.environ:
        os.environ["cfg_name"] = cfg_name
    return main_func(args)


if __name__ == "__main__":
    import time
    import statistics
    from pprint import pprint
    from types import SimpleNamespace

    root_dir = "data"
    task_name = "example"
    task_path = f"{root_dir}/{task_name}"

    # dataset = McapLeRobotDataset(
    #     McapLeRobotDatasetConfig(
    #         data_root=task_path,
    #         states=[
    #             "/follow/arm/joint_state/position",
    #             "/follow/eef/joint_state/position",
    #         ],
    #         images=["/env_camera/color/image_raw"],
    #         actions=["/lead/arm/pose/position", "/lead/arm/pose/orientation"],
    #         horizon=HorizonConfig(fill_with_last=True, future_num=1),
    #     )
    # )
    class Config:
        dataset = SimpleNamespace(root=root_dir, repo_id=task_name)
        policy = SimpleNamespace(action_delta_indices=list(range(2)))

    dataset = make_dataset(Config)
    print(f"{dataset.num_episodes} episodes, {dataset.num_frames} frames")
    pprint(dataset.meta.features)
    # pprint(dataset.meta.stats)
    for key, stat in dataset.meta.stats.items():
        print(f"{key}: {stat['n']}")
    time_costs = []
    start = time.perf_counter()
    for i, data in enumerate(dataset):
        input("Press Enter to continue...")
        time_costs.append(time.perf_counter() - start)
        if i == 0:
            for key, value in data.items():
                print(f"{key}: {value.shape} {value.dtype} {value.device}")
        start = time.perf_counter()
    assert i == dataset.num_frames - 1, (
        f"Expected {dataset.num_frames} frames, but got {i + 1}"
    )
    next(iter(dataset))  # make sure non-iterator
    print(statistics.mean(time_costs[1:]))
