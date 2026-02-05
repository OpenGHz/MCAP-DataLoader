from mcap_data_loader.datasets.mcap_dataset import (
    McapMultiEpisodeDatasets,
    McapMultiEpisodeDatasetsConfig,
)
from mcap_data_loader.utils.hydra_utils import hydra_instance_from_dict
from mcap_data_loader.pipelines.basis import Pipeline, PipelineConfig
from pprint import pprint
from omegaconf import OmegaConf
import logging


state_keys = [
    "/follow/arm/joint_state/position",
    "/follow/eef/joint_state/position",
]
image_keys = ["/env_camera/color/image_raw"]
action_keys = ["/lead/arm/pose/position", "/lead/arm/pose/orientation"]
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
                    "_target_": "mcap_data_loader.pipelines.Horizon",
                    "fill_with_last": True,
                    "future_num": 1,
                },
                3: {
                    "_target_": "mcap_data_loader.callers.Map",
                    "callable": {
                        "_target_": "mcap_data_loader.callers.stack.HorizonStacker",
                        "now": {
                            "observation.state": state_keys,
                            # "observation.effort": "/follow/arm/joint_state/effort",
                            "observation.images.env_camera": image_keys[0],
                        },
                        "future": {
                            "action": action_keys,
                        },
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lerobot_data_loader")
logger.info("Instantiating configuration...")
pipeline_config = hydra_instance_from_dict(pipeline_dict)

print("\n", "*" * 20, "Pipeline Configuration", "*" * 20)
pprint(OmegaConf.to_container(pipeline_config))
print("\n")


logger.info("Creating datasets...")
root_dir = "data/example"
data_root = root_dir
keys = state_keys + image_keys + action_keys
datasets = McapMultiEpisodeDatasets(
    McapMultiEpisodeDatasetsConfig(
        common={"data_root": data_root},
        configs={
            "arm_dataset": {"keys": keys[:2]},
            "eef_dataset": {"keys": keys[2:]},
        },
    )
)

logger.info("Initializing the pipeline...")
pipeline = Pipeline(PipelineConfig(pipeline=pipeline_config))
logger.info("Applying the pipeline...")
data_loader = pipeline(datasets)
logger.info(f"Iterating {data_loader}...")
for sample in data_loader:
    # pprint(sample)
    for key, value in sample.items():
        if isinstance(value, (float, int)):
            print(f"{key:30s} -> {value}")
        else:
            print(f"{key:30s} -> shape {value.shape}, dtype {value.dtype}")
    break
logger.info("Done.")
