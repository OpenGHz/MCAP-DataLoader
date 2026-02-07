from torch.utils.data import IterableDataset
from torch import Tensor, asarray
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Union, Dict
from collections.abc import Mapping
from mcap_data_loader.datasets.mcap_dataset import (
    McapMultiEpisodeDatasets,
    McapMultiEpisodeDatasetsConfig,
)
from mcap_data_loader.utils.hydra_utils import hydra_instance_from_dict
from mcap_data_loader.utils.basic import force_set_attr
from mcap_data_loader.utils.stat import concatenate_statistics, Statistics
from mcap_data_loader.pipelines import Pipeline, PipelineConfig, HorizonConfig


class McapLeRobotDatasetConfig(BaseModel, frozen=True):
    """Configuration for McapLeRobotDataset."""

    model_config = ConfigDict(validate_assignment=True)

    data_root: Union[str, Dict[str, List[str]]]
    """The root directory of the dataset."""
    states: List[str] = []
    """The list of state keys."""
    images: List[str] = []
    """The list of image keys."""
    actions: List[str]
    """The list of action keys."""
    horizon: HorizonConfig = {}

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


class McapLeRobotDatasetMeta(BaseModel, frozen=True):
    """Metadata for McapLeRobotDataset."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    features: Dict[str, Dict[str, Union[tuple, str]]]
    """The dictionary of features, where the key is the feature name and the value is a dictionary containing the shape and dtype of the feature."""
    stats: Dict[str, Statistics]
    """The dictionary of statistics, where the key is the feature name and the value is a dictionary containing the statistics of the feature."""

    @field_validator("stats", mode="after")
    def validate_stats(cls, v: dict):
        for stat in v.values():
            stat["count"] = stat["n"]
        return v


ItemType = Dict[str, Tensor]

STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEY_PREFIX = "observation.images"


class McapLeRobotDataset(IterableDataset):
    def __init__(self, config: McapLeRobotDatasetConfig):
        self.config = config
        pipeline_dict = {
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
                                # "_target_": "mcap_data_loader.utils.dict.valmap_include",
                                "_target_": "mcap_data_loader.basis.DataStamped.map_dict",
                                "_partial_": True,
                                "_args_": [
                                    {
                                        "_target_": "einops.rearrange",
                                        "_partial_": True,
                                        "pattern": "h w c -> c h w",
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
                                "now": {
                                    STATE_KEY: config.states,
                                    # "observation.effort": "/follow/arm/joint_state/effort",
                                }
                                | {
                                    IMAGE_KEY_PREFIX
                                    + img_key.removeprefix("/").split("/")[0]: img_key
                                    for img_key in config.images
                                },
                                "future": {ACTION_KEY: config.actions},
                                # NOTE: the same as the input, usually cpu
                                # "backend_out": "torch",
                                "dtype": "float32",
                            },
                        },
                        4: {
                            "_target_": "torchdata.nodes.IterableWrapper",
                            "_partial_": True,
                        },
                    },
                },
            },
            2: {"_target_": "mcap_data_loader.callers.nodes.MultiNodeWeightedSampler"},
        }
        self._datasets = McapMultiEpisodeDatasets(
            McapMultiEpisodeDatasetsConfig(
                configs={
                    data_root: {"data_root": data_root, "keys": keys}
                    for data_root, keys in config.data_root.items()
                },
            )
        )
        pipeline_config = hydra_instance_from_dict(pipeline_dict)
        pipeline = Pipeline[ItemType](PipelineConfig(pipeline=pipeline_config))
        self._pipeline = pipeline(self._datasets)
        self._ds_iter = None
        first_item = next(iter(self._pipeline))
        self._add_items = {
            "action_is_pad": asarray([True] * first_item[ACTION_KEY].shape[0]),
        }
        # NOTE: do not update the `first_item` here, otherwise the `action_is_pad` will
        # be treated as an action feaute
        # first_item.update(self._add_items)
        # NOTE: the `timestamp` in lerobot dataset is float32, but here
        # it is int64
        features = {}
        for key, value in first_item.items():
            shape = tuple(value.shape)
            if len(value.shape) == 3:
                dtype = "image"
            else:
                dtype = str(value.dtype).split(".")[-1]
            features[key] = {"shape": shape, "dtype": dtype}
        stats = {}
        data_stats = self._datasets.statistics()
        for concat_key, keys in zip(
            (STATE_KEY, ACTION_KEY), (config.states, config.actions)
        ):
            stats[concat_key] = concatenate_statistics(
                [data_stats[key] for key in keys]
            )
        self._meta = McapLeRobotDatasetMeta(features=features, stats=stats)

    def __iter__(self):
        # return iter(self._pipeline)
        # TODO: dynamically adjust the `action_is_pad` shape if not fill_with_last
        for item in self._pipeline:
            item.update(self._add_items)
            yield item

    def __getitem__(self, index):
        if index == 0:
            self._ds_iter = iter(self._pipeline)
        item = next(self._ds_iter)
        item.update(self._add_items)
        return item

    @property
    def meta(self) -> McapLeRobotDatasetMeta:
        return self._meta


if __name__ == "__main__":
    import time
    import statistics
    from pprint import pprint

    root_dir = "data/example"
    dataset = McapLeRobotDataset(
        McapLeRobotDatasetConfig(
            data_root=root_dir,
            states=[
                "/follow/arm/joint_state/position",
                "/follow/eef/joint_state/position",
            ],
            images=["/env_camera/color/image_raw"],
            actions=["/lead/arm/pose/position", "/lead/arm/pose/orientation"],
            horizon=HorizonConfig(fill_with_last=True, future_num=1),
        )
    )
    # pprint(dataset.meta.features)
    pprint(dataset.meta.stats)
    time_costs = []
    start = time.perf_counter()
    for data in dataset:
        time_costs.append(time.perf_counter() - start)
        # for key, value in data.items():
        #     print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
        start = time.perf_counter()

    print(statistics.mean(time_costs[1:]))
