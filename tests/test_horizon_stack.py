from mcap_data_loader.callers.stack import HorizonStacker, HorizonStackerConfig
from pprint import pprint
import numpy as np
import time


item = {
    "/follow/arm/joint_state/position": {"data": [0.0, 0.1], "t": 0.0},
    "/follow/eef/joint_state/position": {"data": [1.0, 1.1], "t": 0.0},
    "/lead/arm/joint_state/position": {"data": [2.0, 2.1], "t": 0.0},
    "/lead/eef/joint_state/position": {"data": [3.0, 3.1], "t": 0.0},
}
for value in item.values():
    value["data"] = np.array(value["data"])
horizon_data = ((item,), (item, item))


config = HorizonStackerConfig(
    # past={
    #     "/past_state": [
    #         "/follow/arm/joint_state/position",
    #         "/follow/eef/joint_state/position",
    #     ]
    # },
    future={
        "action": [
            "/lead/arm/joint_state/position",
            "/lead/eef/joint_state/position",
        ]
    },
    now={
        "observation.state": [
            "/follow/arm/joint_state/position",
            "/follow/eef/joint_state/position",
        ]
    },
    backend_out="torch",
    dtype="float32",
)
stacker = HorizonStacker(config)

# do not count the first call for fair timing, as it includes initialization overhead
stacked = stacker(horizon_data)
start = time.time()
for _ in range(10):
    stacked = stacker(horizon_data)
print(f"Average time per call: {(time.time() - start) / 10 * 1000:.6f} ms")

pprint(stacked)
for k, v in stacked.items():
    if not isinstance(v, float):
        print(f"{k:12s} -> shape {v.shape}, dtype {v.dtype}")
    else:
        print(f"{k:12s} -> {v}")
