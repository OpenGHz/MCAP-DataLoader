from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from pprint import pprint
import argparse
import logging
import json


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
            "arm/joint_state/position",
            "eef/joint_state/position",
            "hand_cam/mask/heat_map",
            "hand_cam/color/image_raw",
        ],
        strict=True,
    )
)

# for index, sample in enumerate(dataset.reader.iter_attachment_samples(color_topics)):
#     # print(f"Sample {index}: {sample.keys()}")
#     # print(index)
#     pass

for index, episode in enumerate(dataset):
    pprint(f"{episode.config.data_root}: {len(episode)} samples")
    for attachment in episode.reader.reader.iter_attachments():
        if attachment.name == "component_info":
            data = json.loads(attachment.data)
            pprint(data)
    for sample in episode:
        print(sample.keys())
        # pprint(sample)
        break
    break
