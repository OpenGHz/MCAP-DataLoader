from airdc.common.samplers.mcap_smaplers.sampler_flb import (
    McapFlbDataSampler,
    McapFlbDataSamplerConfig,
)
from collections import defaultdict
from pathlib import Path
import time


if __name__ == "__main__":
    out_dir = Path("data/example_written")

    out_dir.mkdir(parents=True, exist_ok=True)
    episode = 0

    mcap_sampler = McapFlbDataSampler(McapFlbDataSamplerConfig())

    mcap_sampler.set_info({})

    assert mcap_sampler.configure()

    path = mcap_sampler.compose_path(out_dir, episode)

    # sampling data
    sample_count = 10
    left_data = defaultdict(list)
    for _ in range(sample_count):
        raw_data = {
            "arm/pose/position": [0.1, 0.2, 0.3],
            "arm/pose/orientation": [0.1, 0.2, 0.3, 0.4],
            "arm/pose/rot6d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "arm/pose/position_rela": [0.0, 0.0, 0.0],
            "arm/pose/orientation_rela": [0.0, 0.0, 0.0, 1.0],
            "arm/pose/rot6d_rela": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        sample = {
            key: {"data": value, "t": time.time_ns()} for key, value in raw_data.items()
        }
        sample.update({"log_stamps": time.time_ns()})
        left = mcap_sampler.update(sample)
        for key, value in left.items():
            left_data[key].append(value)

    mcap_sampler.save(path, left_data)

    mcap_sampler.shutdown()

    print(f"Saved {sample_count} samples to {path}")
