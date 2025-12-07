"""Example of creating multi-episode datasets."""

from pprint import pprint


def create_multi_datasets_example():
    """Example of creating multi-episode datasets."""
    from mcap_data_loader.datasets.mcap_dataset import (
        McapMultiEpisodeDatasets,
        McapMultiEpisodeDatasetsConfig,
        DataRearrangeConfig,
        RearrangeType,
    )

    config = McapMultiEpisodeDatasetsConfig(
        common={
            "strict": False,
            "rearrange": DataRearrangeConfig(dataset=RearrangeType.SORT_STEM_DIGITAL),
        },
        configs={
            0: {
                "data_root": "data/example",
                "keys": ["/follow/arm/joint_state/position"],
            },
            1: {
                "data_root": "data/example_copy",
                "keys": {
                    0: "/env_camera/color/image_raw",
                    1: "/env_camera/depth/image_raw",
                },
            },
        },
    )
    pprint(config.model_dump())
    dataset = McapMultiEpisodeDatasets(config)
    return dataset


if __name__ == "__main__":
    dataset = create_multi_datasets_example()
    pprint(dataset.statistics())
    for idx, episodes in enumerate(dataset):
        for episode in episodes:
            pprint(
                f"Dataset {idx} in {episode.config.data_root}: {len(episode)} samples"
            )
