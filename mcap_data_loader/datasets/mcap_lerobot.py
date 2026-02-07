from torch.utils.data import IterableDataset
from torch import Tensor
from pydantic import BaseModel, field_validator
from typing import List, Union, Dict
from collections.abc import Mapping, Iterator
from mcap_data_loader.datasets.mcap_dataset import (
    McapMultiEpisodeDatasets,
    McapMultiEpisodeDatasetsConfig,
)
from mcap_data_loader.utils.hydra_utils import hydra_instance_from_dict
from mcap_data_loader.pipelines import Pipeline, PipelineConfig, HorizonConfig


class McapLeRobotDatasetConfig(BaseModel):
    """Configuration for McapLeRobotDataset."""

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

    def model_post_init(self, context):
        if isinstance(self.data_root, str):
            self.data_root = {self.data_root: self.states + self.images + self.actions}


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
                            "fill_with_last": True,
                            "future_num": 1,
                        },
                        3: {
                            "_target_": "mcap_data_loader.callers.Map",
                            "callable": {
                                "_target_": "mcap_data_loader.callers.stack.HorizonStacker",
                                "now": {
                                    "observation.state": config.states,
                                    # "observation.effort": "/follow/arm/joint_state/effort",
                                }
                                | {
                                    "observation.images."
                                    + img_key.removeprefix("/").split("/")[0]: img_key
                                    for img_key in config.images
                                },
                                "future": {"action": config.actions},
                                "backend_out": "torch",
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
        pipeline = Pipeline(PipelineConfig(pipeline=pipeline_config))
        self._pipeline = pipeline(self._datasets)
        self._ds_iter = None

    def __iter__(self) -> Iterator[Dict[str, Union[int, Tensor]]]:
        return iter(self._pipeline)

    def __getitem__(self, index):
        if index == 0:
            self._ds_iter = iter(self._pipeline)
        return next(self._ds_iter)


if __name__ == "__main__":
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
        )
    )
    for data in dataset:
        # print(data)
        for key, value in data.items():
            if isinstance(value, int):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
        break
