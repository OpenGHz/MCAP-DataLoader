"""
Example of using PyTorch data nodes to create a data loading and processing pipeline.
"""

if __name__ == "__main__":
    from torchdata import nodes
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )
    from pprint import pprint, pformat
    import numpy as np

    root_dir = "data/example"
    data_root = root_dir
    keys = [
        "/follow/arm/joint_state/position",
        "/follow/eef/joint_state/position",
    ]
    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys)
    )
    dataset.load()

    source_nodes = {}
    weights = {}
    for episode in dataset:
        source_nodes[episode.config.data_root] = nodes.IterableWrapper(episode)
        weights[episode.config.data_root] = 1.0

    pprint(weights)

    node = nodes.MultiNodeWeightedSampler(
        source_nodes,
        weights,
        nodes.StopCriteria.ALL_DATASETS_EXHAUSTED,
    )

    def process_sample(sample):
        return np.concatenate(
            [
                sample["/follow/arm/joint_state/position"],
                sample["/follow/eef/joint_state/position"],
            ]
        )

    node = nodes.ParallelMapper(node, process_sample, 0)
    node = nodes.Batcher(node, 2, False)
    node = nodes.Prefetcher(node, 10, 1000)
    # node = nodes.PinMemory(node, "cuda", 1000)
    # node = nodes.Unbatcher(node)
    node = nodes.Loader(node, False)

    max_steps = 10
    for idx, data in enumerate(node):
        print(f"Step {idx}: \n {pformat(data)}")
        if idx + 1 >= max_steps:
            break
