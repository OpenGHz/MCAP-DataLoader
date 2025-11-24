if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapMultiEpisodeDatasets,
        McapMultiEpisodeDatasetsConfig,
        DataRearrangeConfig,
        RearrangeType,
    )
    from pprint import pprint

    config = McapMultiEpisodeDatasetsConfig(
        common={
            "strict": False,
            "rearrange": DataRearrangeConfig(episode=RearrangeType.SORT_STEM_DIGITAL),
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
    for idx, episodes in enumerate(dataset):
        for episode in episodes:
            pprint(
                f"Dataset {idx} in {episode.config.data_root}: {len(episode)} samples"
            )
