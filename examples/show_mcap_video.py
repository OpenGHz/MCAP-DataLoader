from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
    DataRearrangeConfig,
    RearrangeType,
)
from mcap_data_loader.serialization.video.pyav import DecodeConfig
from pathlib import Path
import argparse
import logging
import numpy as np
import cv2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    type=str,
    help="path to mcap files",
)
parser.add_argument(
    "--imshow",
    action="store_true",
    help="whether to show video frames using cv2.imshow",
)
args = parser.parse_args()
path = args.path

dataset = McapFlatBuffersEpisodeDataset(
    McapFlatBuffersEpisodeDatasetConfig(
        data_root=path,
        rearrange=DataRearrangeConfig(dataset=RearrangeType.SORT),
        media_configs=[DecodeConfig(mismatch_tolerance=5)],
        with_step=False,
    )
)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


for index in range(1):
    episode = dataset[index]
    logger.info(f"Episode {index}: {episode.config.data_root}")
    ep_reader = episode.reader
    all_attachments = ep_reader.all_attachment_names()
    color_topics = [att for att in all_attachments if "color" in att]
    # re-configure dataset to load color keys
    episode.config.keys.update(color_topics)
    for sample in episode:
        # print(sample)
        images = []
        for key, value in sample.items():
            img = value["data"]
            # print(f"Key: {key}, Image shape: {img.shape}, dtype: {img.dtype}")
            images.append(img)
            # t = value["t"]
            if args.imshow:
                cv2.imshow(key, value["data"])
            if cv2.waitKey(0) in [27, ord("q")]:
                break
        if not args.imshow:
            cv2.imwrite(output_dir / "checked_image.jpg", np.hstack(images))
            if input("Press Enter to continue to next episode, or any key to quit: "):
                break
    if args.imshow:
        logger.info("Press any key to continue to next episode, or 'q'/'ESC' to quit")
        if cv2.waitKey(0) in [27, ord("q")]:
            break

if args.imshow:
    cv2.destroyAllWindows()
