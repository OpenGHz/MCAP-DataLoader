from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from pprint import pprint


path = "data/example"
dataset = McapFlatBuffersEpisodeDataset(
    McapFlatBuffersEpisodeDatasetConfig(
        data_root=path,
        keys=[
            "/follow/arm/joint_state/position",
            "/follow/eef/joint_state/position",
            "/env_camera/color/image_raw",
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
        # pprint(sample)
        # break
        pass
