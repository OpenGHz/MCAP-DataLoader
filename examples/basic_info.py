"""
Get basic information about the dataset.
"""

if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )
    from pprint import pprint

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root="data/example",
            keys=["/follow/arm/joint_state/position", "log_stamps"],
        )
    )
    print(f"All files: {dataset.all_files}")
    print(f"Dataset length: {len(dataset)}")
    print("Dataset statistics:")
    pprint(dataset.statistics())
    for episode in dataset:
        print(f"Current file: {episode.config.data_root}")
        for sample in episode:
            print(f"Sample keys: {sample.keys()}")
            break
        print(f"Episode length: {len(episode)}")
        print(f"All topics: {episode.reader.all_topic_names()}")
        print(f"All attachments: {episode.reader.all_attachment_names()}")
        print("Episode statistics:")
        pprint(episode.statistics())
        print("----" * 10)
