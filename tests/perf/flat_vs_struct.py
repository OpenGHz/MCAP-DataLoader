from mcap_data_loader.serialization.flb import McapFlatBuffersReader
from pathlib import Path
from time import perf_counter
import numpy as np


path = Path("data/perf/flat_struct.mcap")
assert path.exists(), f"Data file does not exist: {path}"

with open(path, "rb") as f:
    reader = McapFlatBuffersReader(f)

    times = []
    start = perf_counter()
    keys = ["/right/lead/eef/pose/position", "/right/lead/eef/pose/orientation"]
    for i, sample in enumerate(reader.iter_message_samples(keys)):
        for key in keys:
            field_value = sample[key]["data"]
        # print(sample)

    end = perf_counter()
    print(f"Time taken to read {i + 1} samples: {end - start:.4f} seconds")

    key = "/right/lead/arm/pose"
    for i, sample in enumerate(reader.iter_message_samples([key])):
        pose = sample[key]["data"]["pose"]
        pos, ori = pose["position"], pose["orientation"]
        position = np.array([pos["x"], pos["y"], pos["z"]])
        orientation = np.array([ori["x"], ori["y"], ori["z"], ori["w"]])
        # print(sample)

    end = perf_counter()
    print(f"Time taken to read {i + 1} samples: {end - start:.4f} seconds")
