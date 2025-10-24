"""
Flatten Pipeline
"""

if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )
    from mcap_data_loader.pipelines.flatten import Flatten, FlattenConfig

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root="data/example",
            keys=["/follow/arm/joint_state/position", "log_stamps"],
        )
    )
    dataset.load()

    for sample in Flatten(FlattenConfig())(dataset):
        print(sample)
        break
