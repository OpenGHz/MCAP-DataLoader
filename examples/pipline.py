if __name__ == "__main__":
    from mcap_data_loader.piplines import NestedZipConfig, NestedZip, Merge, MergeConfig
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )
    from pprint import pprint

    root_dir = "data/example"
    data_root = root_dir
    keys = [
        "/follow/arm/joint_state/position",
        "/follow/eef/joint_state/position",
    ]
    datasets = (
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys[:1])
        ),
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys[1:])
        ),
    )
    for dataset in datasets:
        dataset.load()
    nested = NestedZip(NestedZipConfig(depth=1))(datasets)
    for episodes in nested:
        merged = Merge[dict](MergeConfig(method="auto"))(episodes)
        for sample in merged:
            pprint(sample)
            assert set(sample.keys()) == set(keys)
            break
        break
