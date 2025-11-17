"""
Reading MCAP files concurrently example.
"""

if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
        McapFlatBuffersSampleDataset,
    )
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root="data/example",
            keys=["/follow/arm/joint_state/position", "log_stamps"],
        )
    )

    def read_episode(episode: McapFlatBuffersSampleDataset):
        root = episode.config.data_root
        print(f"Reading episode from {root} with {len(episode)} samples")
        for index, sample in enumerate(episode):
            print(sample.keys())
            time.sleep(0.5)  # simulate some processing time
            if index >= 3:
                break
        return root

    futures = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        for episode in dataset:
            futures.append(executor.submit(read_episode, episode))

        for future in as_completed(futures):
            print(f"Finished reading episode from {future.result()}")

    print("All episodes have been read.")
