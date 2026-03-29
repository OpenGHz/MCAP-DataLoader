from torch.utils.data import IterableDataset
from torch import Tensor, asarray
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Union, Dict
from functools import cached_property
from collections.abc import Mapping
import random
from queue import Queue, Empty, Full
from threading import Thread, Event
from mcap_data_loader.datasets.mcap_dataset import (
    McapMultiEpisodeDatasets,
    McapMultiEpisodeDatasetsConfig,
)
from mcap_data_loader.utils.hydra_utils import hydra_instance_from_dict
from mcap_data_loader.utils.basic import force_set_attr
from mcap_data_loader.utils.stat import concatenate_statistics, Statistics
from mcap_data_loader.pipelines import Pipeline, PipelineConfig, HorizonConfig
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
    shuffle_episodes: bool = True
    """Whether to shuffle the merged episode order for each new pass."""
    shuffle_seed: int = 0
    """Base random seed for episode-level shuffling."""
    prefetch_items: int = 8
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


class McapLeRobotDataset(IterableDataset):
    def __init__(self, config: McapLeRobotDatasetConfig):
        self.config = config
        self._camera_mappings = {
            IMAGE_KEY_PREFIX + "." + img_key.removeprefix("/").split("/")[0]: img_key
            for img_key in config.images
        }
        self._episode_pipeline_dict = {
            0: {
                "_target_": "mcap_data_loader.pipelines.NestedZip",
                "depth": 1,
            },
            1: {
                "_target_": "mcap_data_loader.callers.Map",
                "callable": {
                    "_target_": "mcap_data_loader.pipelines.Pipeline",
                    "pipeline": {
                        0: {
                            "_target_": "mcap_data_loader.pipelines.Merge",
                            "replace": True,
                        },
                        1: {
                            "_target_": "mcap_data_loader.callers.Map",
                            "callable": {
                                # "_target_": "mcap_data_loader.callers.chain.CallerChain",
                                # "_target_": "mcap_data_loader.utils.dict.valmap_include",
                                "_target_": "mcap_data_loader.basis.DataStamped.map_dict",
                                "_partial_": True,
                                "_args_": [
                                    {
                                        # "_target_": "einops.rearrange",
                                        # "pattern": "h w c -> c h w",
                                        "_target_": "mcap_data_loader.utils.array_like.rearrange_and_shrink_np",
                                        "transpose": (2, 0, 1),
                                        "factor": 255.0,
                                        "dtype": "float32",
                                        "_partial_": True,
                                    }
                                ],
                                "keys": config.images,
                            },
                        },
                        2: {
                            "_target_": "mcap_data_loader.pipelines.Horizon",
                            **config.horizon.model_dump(),
                        },
                        3: {
                            "_target_": "mcap_data_loader.callers.Map",
                            "callable": {
                                "_target_": "mcap_data_loader.callers.stack.HorizonStacker",
                                "now": (
                                    {
                                        STATE_KEY: config.states,
                                        # "observation.effort": "/follow/arm/joint_state/effort",
                                    }
                                    # TODO: empty keys may should not be considered as stackable
                                    if config.states
                                    else {}
                                )
                                | self._camera_mappings,
                                "future": {ACTION_KEY: config.actions},
                                "backend_out": "torch",
                                "dtype": "float32",
                                # must be cpu: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
                                "device": "cpu",
                            },
                        },
                    },
                },
            },
        }

        self._datasets = self._make_datasets()
        self._ds_iter = None
        self._epoch = 0
        first_item = next(self._iter_items(self._make_datasets(), shuffle=False))
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
                    "media_configs": [DecodeConfig(frame_format="rgb24")],
                },
                configs={
                    data_root: {"data_root": data_root, "keys": keys}
                    for data_root, keys in self.config.data_root.items()
                },
            )
        )

    def _make_episode_pipeline(self, datasets: McapMultiEpisodeDatasets):
        pipeline_config = hydra_instance_from_dict(self._episode_pipeline_dict)
        pipeline = Pipeline[ItemType](PipelineConfig(pipeline=pipeline_config))
        return pipeline(datasets)

    def _iter_items(
        self, datasets: McapMultiEpisodeDatasets, shuffle: bool | None = None
    ):
        # Iterate merged episodes sequentially so that only one episode stream is
        # active at a time. The old MultiNodeWeightedSampler interleaved hundreds
        # of episode iterators, which kept many image decoders/buffers alive and
        # made memory grow with the number of partially-consumed episodes.
        episode_iters = list(self._make_episode_pipeline(datasets))
        if shuffle is None:
            shuffle = self.config.shuffle_episodes
        if shuffle and len(episode_iters) > 1:
            random.Random(self.config.shuffle_seed + self._epoch).shuffle(episode_iters)
        for episode_iter in episode_iters:
            yield from episode_iter

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
        if self.config.prefetch_items > 0:
            item_iter = self._iter_prefetched(item_iter)
        return item_iter

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


def train():
    from lerobot.scripts import lerobot_train
    from mcap_data_loader.datasets.mcap_lerobot import make_dataset
    from mcap_data_loader.utils.cli import extract_and_remove_args, extend_args
    from functools import partial
    from mcap_data_loader.scripts.run_with_yaml import parse_args, get_args_list
    import sys

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
    # print(sys.argv), exit(0)
    return lerobot_train.main()


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
