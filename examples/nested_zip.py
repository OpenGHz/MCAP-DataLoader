if __name__ == "__main__":
    from mcap_data_loader.piplines import NestedZipConfig, NestedZip
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
        McapFlatBuffersSampleDataset,
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
    depth = 2
    nested = NestedZip(NestedZipConfig(depth=depth))(datasets)
    for items in nested:
        pprint(items)
        assert len(items) == len(datasets)
        if depth == 0:
            assert isinstance(items[0], McapFlatBuffersEpisodeDataset)
        elif depth == 1:
            assert isinstance(items[0], McapFlatBuffersSampleDataset)
        elif depth == 2:
            assert isinstance(items[0], dict)
        elif depth == 3:
            assert isinstance(items[0], str)
        else:
            raise ValueError(f"Unsupported depth {depth}.")
        break
