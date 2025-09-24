from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
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
        keys=[],
        topics=[],
        attachments=None,
    )
)
dataset.load()

# TODO: make episode a sample class?
# TODO: improve dynamic iteration
for index, episode in enumerate(dataset):
    logger.info(f"Episode {index}: {dataset.current_file}")
    ep_reader = dataset.reader[dataset.current_file]
    all_attachments = ep_reader.all_attachment_names()
    color_topics = [att for att in all_attachments if "color" in att]
    for index, sample in enumerate(ep_reader.iter_attachment_samples(color_topics)):
        for key, image in sample.items():
            cv2.imshow(key, image)
        if cv2.waitKey(0) in [27, ord("q")]:
            break
    logger.info("Press any key to continue to next episode, or 'q'/'ESC' to quit")
    if cv2.waitKey(0) in [27, ord("q")]:
        break
cv2.destroyAllWindows()
