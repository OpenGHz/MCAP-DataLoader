if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapMultiEpisodeDataset,
        McapMultiEpisodeDatasetConfig,
        DataRearrangeConfig,
        RearrangeType,
    )
    from pprint import pprint

    config = McapMultiEpisodeDatasetConfig(
        common={
            "strict": False,
            "rearrange": DataRearrangeConfig(episode=RearrangeType.SORT_STEM_DIGITAL),
        },
        roots={
            "data/example": ["/follow/arm/joint_state/position"],
            "data/example_copy": ["/env_camera/color/image_raw"],
            # add other keys to data/example with a `ID//` prefix to avoid collision
            # this is useful for incremental configuration adjustment
            "0//data/example": ["/follow/eef/joint_state/position"],
        },
    )
    pprint(config.model_dump())
    dataset = McapMultiEpisodeDataset(config)
    for idx, episodes in enumerate(dataset):
        for episode in episodes:
            pprint(
                f"Dataset {idx} in {episode.config.data_root}: {len(episode)} samples"
            )
