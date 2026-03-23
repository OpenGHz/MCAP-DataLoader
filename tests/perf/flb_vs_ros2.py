from mcap_data_loader.serialization.ros import McapROSReader
from mcap_data_loader.serialization.flb import McapFlatBuffersReader
from pathlib import Path
from time import perf_counter
import numpy as np


path = Path("data/perf/ros/0.mcap")
assert path.exists(), f"Data file does not exist: {path}"

with open(path, "rb") as f:
    reader = McapROSReader(f)

    times = []
    start = perf_counter()
    key = "/robot/arm_right_lead/poses"
    for i, sample in enumerate(reader.iter_message_samples([key])):
        # print(sample)
        pos = sample[key]["data"].pose.position
        position = np.array([pos.x, pos.y, pos.z])
    end = perf_counter()
    print(f"Time taken to read {i+1} samples: {end - start:.4f} seconds")

path = Path("data/perf/flb/0.mcap")
assert path.exists(), f"Data file does not exist: {path}"

with open(path, "rb") as f:
    reader = McapFlatBuffersReader(f)

    times = []
    start = perf_counter()
    key = "/right/lead/eef/pose/position"
    for i, sample in enumerate(reader.iter_message_samples([key])):
        position = sample[key]["data"]
        # print(sample)
    end = perf_counter()
    print(f"Time taken to read {i+1} samples: {end - start:.4f} seconds")
