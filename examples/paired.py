"""
Example of using `more_itertools.pairwise` to process paired data.
This is particularly useful for scenarios where you need to compare
consecutive samples, such as in time series analysis or when calculating
differences between successive data points.
"""

if __name__ == "__main__":
    import time
    import logging
    from more_itertools import pairwise
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger("paired")

    root_dir = "data/example"
    data_root = root_dir
    keys = [
        "/follow/arm/joint_state/position",
        "/follow/eef/joint_state/position",
    ]

    last_number = 2
    next_number = 20

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys)
    )
    dataset.load()
    start = time.perf_counter()
    for episode in dataset:
        start = time.perf_counter()
        for step, pair in enumerate(pairwise(episode)):
            assert len(pair) == 2
            print(f"{step=}", pair[0].keys(), pair[1].keys())
        else:
            print(
                f"Processed {len(episode)} samples in episode {episode.config.data_root}"
            )
        total_time = time.perf_counter() - start
        avg_time = total_time / 2
        print(f"Average time per sample: {avg_time:.5f} seconds")
        print(f"Total time taken: {total_time:.5f} seconds")
        break  # Only process the first episode
