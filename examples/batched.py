"""
Example of using `more_itertools.batched` to process data in batches.
This is particularly useful for scenarios like training machine learning models
where batch processing can improve efficiency.
"""

if __name__ == "__main__":
    import time
    from more_itertools import batched
    import logging
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )

    logging.basicConfig(level=logging.INFO)

    root_dir = "data/example"
    data_root = root_dir
    keys = [
        "/follow/arm/joint_state/position",
        "/follow/eef/joint_state/position",
    ]

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys)
    )
    batch_size = 10
    start = time.perf_counter()
    for episode in dataset:
        start = time.perf_counter()
        try:
            for step, batch in enumerate(batched(episode, batch_size, strict=True)):
                print(f"{step=}", batch[0].keys())
                assert len(batch) == batch_size
            else:
                print(
                    f"Processed {len(episode)} samples in episode {episode.config.data_root}"
                )
        except ValueError as e:
            if "batched()" in str(e):
                print(f"Could not form a complete batch of size {batch_size}")
            else:
                raise e
        total_time = time.perf_counter() - start
        avg_time = total_time / batch_size
        print(f"Average time per sample: {avg_time:.5f} seconds")
        print(f"Total time taken for {batch_size=}: {total_time:.5f} seconds")
        break  # Only process the first episode
