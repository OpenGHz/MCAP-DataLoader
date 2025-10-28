"""
Example of rearranging data files in an episode dataset.
"""

if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
        DataRearrangeConfig,
        RearrangeType,
    )

    for re in RearrangeType:
        dataset = McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(
                data_root="data/example",
                keys=["/follow/arm/joint_state/position", "log_stamps"],
                rearrange=DataRearrangeConfig(dataset=re),
            )
        )
        dataset.load()
        print(re, dataset.all_files)
