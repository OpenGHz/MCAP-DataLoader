from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from mcap_data_loader.utils.rela_abs import PoseGlobalRelaAbsTool
from mcap_data_loader.utils.rot6d import Rotation6D
from pprint import pprint
# from airdc.common.samplers.mcap_sampler import McapDataSampler, McapDataSamplerConfig
from airdc.common.samplers.mcap_samplers.sampler_flb import McapFlbDataSampler, McapFlbDataSamplerConfig
from collections import defaultdict
from pathlib import Path
from typing import List, Set
import argparse
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    type=str,
    help="path to mcap files",
)
parser.add_argument(
    "--keys",
    type=str,
    nargs="+",
    help="keys to load from mcap files",
)
parser.add_argument(
    "--targets",
    nargs="+",
    choices=("rela", "rotation_6d"),
    default=("rela", "rotation_6d"),
    help="Derived targets to generate.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    help="Directory to save the processed MCAP files.",
)
args = parser.parse_args()
path = Path(args.path)
keys = args.keys
targets = set(args.targets)
out_dir = args.out_dir

out_dir = path.parent / f"{path.name}_processed" if out_dir is None else Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

dataset = McapFlatBuffersEpisodeDataset(
    McapFlatBuffersEpisodeDatasetConfig(
        data_root=path,
        keys=keys or [],
        with_step=False,
        strict=True,
    )
)


def extract_keys(keys: List[str]) -> Set[str]:
    key_set = set()
    for key in keys:
        if (
            key.endswith("_rela")
            or key.endswith("position")
            or key.endswith("orientation")
            or key.endswith("rotation_6d")
        ):
            key_set.add(key)
    return key_set


if not keys:
    keys = extract_keys(dataset.statistics().keys())
    keys.add("log_stamps")
    dataset.config.keys.update(keys)
    dataset.refresh_config()

print(f"Extracted keys: {keys}")

mcap_sampler = McapFlbDataSampler(McapFlbDataSamplerConfig())
mcap_sampler.set_info({})
assert mcap_sampler.configure()
rela = PoseGlobalRelaAbsTool()


for index, episode in enumerate(dataset):
    pprint(f"{episode.config.data_root}: {len(episode)} samples")
    save_path = mcap_sampler.compose_path(out_dir, int(episode.config.data_root.stem))
    init = episode[0]
    print(f"[Episode {index}] Initial keys: {list(init.keys())}")

    left_data = defaultdict(list)
    for sample in episode:
        processed = {}
        processed_t = {}
        for key in keys:
            value = sample[key]
            if key == "log_stamps":
                continue
            value_data = value["data"]
            value_t = value["t"]
            if "rela" in targets:
                if key.startswith("action/"):
                    obs_key = key.removeprefix("action/")
                    init_data = init[obs_key]["data"]
                else:
                    init_data = init[key]["data"]
                if key.endswith("_rela"):
                    value_rela = value_data
                else:
                    if key.endswith("position"):
                        value_rela = rela.to_rela_position(value_data, init_data)
                        assert np.allclose(
                            rela.to_abs_position(value_rela, init_data), value_data
                        )
                    elif key.endswith("orientation"):
                        value_rela = rela.to_rela_orientation(value_data, init_data)
                        assert np.allclose(
                            rela.to_abs_orientation(value_rela, init_data), value_data
                        )
                    elif key.endswith("rotation_6d"):
                        value_rela = rela.to_rela_rot6d(value_data, init_data)
                        assert np.allclose(
                            rela.to_abs_rot6d(value_rela, init_data), value_data
                        )

                    rela_key = f"{key}_rela"
                    processed[rela_key] = value_rela
                    processed_t[rela_key] = value_t
            if "rotation_6d" in targets:
                key_path = Path(key)
                field_name = key_path.name
                if field_name in {"orientation", "orientation_rela"}:
                    rotation_6d_key = f"{str(key_path.parent)}/rotation_6d"
                    rotation_6d_abs_key = rotation_6d_key
                    if key.endswith("_rela"):
                        rotation_6d_key += "_rela"
                    if rotation_6d_key not in sample:
                        processed[rotation_6d_key] = Rotation6D.quat_to_rot6d(
                            value_data
                        )
                        processed_t[rotation_6d_key] = value_t
                        if (
                            "rela" in targets
                            and (rotation_6d_abs_key not in sample)
                            and not key.endswith("_rela")
                        ):
                            rotation_6d_rela_key = (
                                f"{str(key_path.parent)}/rotation_6d_rela"
                            )
                            processed[rotation_6d_rela_key] = Rotation6D.quat_to_rot6d(
                                value_rela
                            )
                            processed_t[rotation_6d_rela_key] = value_t
        processed = {
            key: {"data": value, "t": processed_t[key]}
            for key, value in processed.items()
        }
        processed.update({"log_stamps": sample["log_stamps"]})
        left = mcap_sampler.update(processed)
        for key, value in left.items():
            left_data[key].append(value)
    mcap_sampler.save(save_path, left_data)
    print(f"Saved processed episode to {save_path}")

mcap_sampler.shutdown()
