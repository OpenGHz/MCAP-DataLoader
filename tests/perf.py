if __name__ == "__main__":
    from pprint import pprint
    import time
    from more_itertools import batched
    import logging
    from mcap_data_loader.datasets.dataset import DataSlicesConfig
    import numpy as np
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
        DataRearrangeConfig,
    )

    logging.basicConfig(level=logging.INFO)

    root_dir = "data/arm1-001"
    # data_root = "0.mcap"
    data_root = root_dir
    keys = [
        "/left/follow/arm/joint_state/position",
        "/left/follow/eef/joint_state/position",
        "/left/lead/arm/joint_state/position",
        "/left/lead/eef/joint_state/position",
        "/env_camera/env/color/image_raw",
    ]
    # keys = (
    #     [
    #         # "/follow/arm/joint_state/position",
    #         # "/follow/eef/joint_state/position",
    #     ]
    #     + [
    #         "/env_camera/color/image_raw",
    #         # "/follow_camera/color/image_raw",
    #         # discoverse camera keys
    #         # "/cam_0/color/image_raw",
    #         # "/cam_1/color/image_raw",
    #         "log_stamps",
    #     ]
    # )

    # dataset = McapFlatBuffersDataset(
    #     McapFlatBuffersDatasetConfig(
    #         data_root=data_root,
    #         keys=keys,
    #     )
    # )
    # start = time.perf_counter()
    # for sample in dataset:
    #     print(time.perf_counter() - start)
    #     # pprint(sample)
    #     start = time.perf_counter()
    #     # break  # Only print the first sample

    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root=data_root,
            keys=keys,
            slices=DataSlicesConfig(dataset={root_dir: (0, 1)}),
            rearrange=DataRearrangeConfig(
                episode="sort",
            ),
            cache=True,
        )
    )
    dataset.load()
    print(dataset.all_files)
    print(f"Dataset length: {len(dataset)}")
    pprint(dataset[0].keys())
    for v1, v2 in zip(dataset[0].values(), dataset[0].values()):
        assert np.array_equal(v1, v2), f"{v1=} != {v2=}"
    for v1, v2 in zip(dataset[0].values(), dataset[1].values()):
        if not np.array_equal(v1, v2):
            print("OK: Samples are not equal")
            break
    else:
        raise ValueError("Samples are equal")

    for file_path, reader in dataset.reader.items():
        print(f"File: {file_path}, Messages: {len(reader)}")
    start = time.perf_counter()
    batch_size = 10
    steps = 1
    for episode in dataset:
        next(episode)  # Skip the first sample
        start = time.perf_counter()
        for step, batch in enumerate(batched(episode, batch_size, strict=True)):
            print(f"{step=}", batch[0].keys())
            if step + 1 >= steps:
                break
        else:
            print(f"Processed {len(episode)} samples in episode {dataset.current_file}")
        total_time = time.perf_counter() - start
        avg_time = total_time / batch_size
        print(f"Average time per sample: {avg_time:.5f} seconds")
        print(f"Total time taken for {batch_size=}: {total_time:.5f} seconds")
        break  # Only process the first episode
