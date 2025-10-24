"""
Get episode by index.
"""

if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root="data/example",
            keys=["/follow/arm/joint_state/position", "log_stamps"],
        )
    )
    dataset.load()

    for i in range(2):
        episode = dataset[i]
        print(f"Index: {i} " + "----" * 10)
        print(f"Current file: {episode.config.data_root}")
        print(f"Episode length: {len(episode)}")
