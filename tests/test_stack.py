import numpy as np
import torch
import timeit
from pprint import pprint
from mcap_data_loader.callers.stack import BatchStackerConfig, BatchStacker


def make_sample(i):
    return {
        "0.0key1": {"data": np.ones(3) * (i + 0.0)},
        "0.0key2": {"data": np.ones(2) * (i + 0.1)},
        "1.0key1": {"data": np.ones(3) * (i + 1.0)},
        "1.1key1": {"data": np.ones(3) * (i + 1.0)},
        "1.0key2": {"data": np.ones(2) * (i + 1.1)},
        "1.1key2": {"data": np.ones(2) * (i + 1.1)},
        "1.2key1": {"data": np.ones(3) * (i + 1.2)},
        "1.2key2": {"data": np.ones(2) * (i + 1.3)},
        "0.1key1": {"data": np.ones(3) * (i + 0.1)},
        "0.1key2": {"data": np.ones(2) * (i + 0.2)},
        "1.5key1": {"data": np.ones(3) * (i + 1.5)},
        "1.5key2": {"data": np.ones(2) * (i + 1.6)},
        "meta": {"data": f"sample-{i}"},
    }


batched_samples = [make_sample(i) for i in range(3)]

stack_config = {
    "cur_state": ["0.0key1", "0.0key2"],  # flat style
    "next_action": [["key1", "key2"], [1.0, 1.2, 1]],  # range style
    "complex": [  # complex style
        ["0.0key1", "0.0key2"],
        ["0.1key1", "0.1key2"],
        ["1.0key1", "1.0key2"],
        ["1.5key1", "1.5key2"],
    ],
}


config = BatchStackerConfig(stack=stack_config, backend_out="torch")
print(config)
processor = BatchStacker(config)
processor.configure()
pprint(processor.config.stack)

batched = processor(batched_samples)
# ===========================================================
cost = timeit.timeit(lambda: processor(batched_samples), number=1000)
print(f"Total time for 1000 runs: {cost:.6f} seconds")
print(f"Average time per run: {cost:.6f} ms")

print("=== 输出结果 ===")
for k, v in batched.items():
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(f"{k:12s} -> shape {v.shape}, dtype {v.dtype}, device {v.device}")
    else:
        print(f"{k:12s} -> {v}")
    # print(f"{k:12s} -> {v}")
