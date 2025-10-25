from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
    DataRearrangeConfig,
    RearrangeType,
)
from mcap_data_loader.utils.av_coder import DecodeConfig
import cv2
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
        rearrange=DataRearrangeConfig(dataset=RearrangeType.SORT),
        media_configs=[DecodeConfig(mismatch_tolerance=5)],
    )
)
dataset.load()

for index in range(1):
    episode = dataset[index]
    logger.info(f"Episode {index}: {episode.config.data_root}")
    ep_reader = episode.reader
    all_attachments = ep_reader.all_attachment_names()
    color_topics = [att for att in all_attachments if "color" in att]
    # re-configure dataset to load color keys
    episode.config.keys = color_topics
    for sample in episode:
        # print(sample)
        for key, value in sample.items():
            # t = value["t"]
            cv2.imshow(key, value["data"])
        if cv2.waitKey(0) in [27, ord("q")]:
            break
    logger.info("Press any key to continue to next episode, or 'q'/'ESC' to quit")
    if cv2.waitKey(0) in [27, ord("q")]:
        break
cv2.destroyAllWindows()
