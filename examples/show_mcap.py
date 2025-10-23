from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from pprint import pprint
import argparse
import logging


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    type=str,
    help="path to mcap files",
)
args = parser.parse_args()
path = args.path

dataset = McapFlatBuffersEpisodeDataset(
    McapFlatBuffersEpisodeDatasetConfig(
        data_root=path,
        keys=[
            "/follow/arm/joint_state/position",
            "/follow/eef/joint_state/position",
            # "/env_camera/color/image_raw",
        ],
        strict=False,
    )
)
dataset.load()

# for index, sample in enumerate(dataset.reader.iter_attachment_samples(color_topics)):
#     # print(f"Sample {index}: {sample.keys()}")
#     # print(index)
#     pass

for index, episode in enumerate(dataset):
    pprint(f"{episode.config.data_root}: {len(episode)} samples")
    for sample in episode:
        pprint(sample)
        break
    break
