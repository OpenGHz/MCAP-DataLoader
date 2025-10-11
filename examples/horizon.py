"""
Example of using `extra_itertools.past_future` to create horizons of past and future data.
This is particularly useful for scenarios like time series forecasting,
where you need to consider both past and future contexts. And it is also
useful for RL/IL policy training where an agent needs to predict future action
chunks based on past observations.
"""

if __name__ == "__main__":
    import time
    import logging
    from mcap_data_loader.utils.extra_itertools import past_future
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

    past_num = 2
    future_num = 20

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys)
    )
    dataset.load()
    start = time.perf_counter()
    for episode in dataset:
        start = time.perf_counter()
        for step, horizons in enumerate(
            past_future(episode, past_num, future_num, None, 1, True)
        ):
            assert len(horizons) == 2
            assert len(horizons[0]) == past_num + 1
            assert len(horizons[1]) == future_num + 1
            assert horizons[0][-1] == horizons[1][0]
            print(f"{step=}", horizons[0][0].keys(), horizons[1][0].keys())
        else:
            print(
                f"Processed {len(episode)} samples in episode {episode.config.data_root}"
            )

        total_time = time.perf_counter() - start
        sample_num = past_num + future_num + 1
        avg_time = total_time / sample_num
        print(f"Average time per sample: {avg_time:.5f} seconds")
        print(f"Total time taken for {sample_num=}: {total_time:.5f} seconds")
        break  # Only process the first episode
