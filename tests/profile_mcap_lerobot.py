#!/DATA/disk1/haizhou/.venv/bin/python
"""Small profiler for the MCAP LeRobot dataset adapter.

This focuses on the current custom MCAP path and prints a cProfile breakdown so we
can see whether time is dominated by image conversion, video decoding, or sample
assembly on a real dataset/config.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

from mcap_data_loader.datasets.mcap_lerobot import (
    McapLeRobotDataset,
    McapLeRobotDatasetConfig,
)
from mcap_data_loader.pipelines import HorizonConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="+", help="One or more MCAP dataset roots.")
    parser.add_argument("--states", nargs="*", default=["arm/pose/position_rela", "arm/pose/rotation_6d_rela"])
    parser.add_argument("--images", nargs="*", default=["hand_cam/color/image_raw"])
    parser.add_argument("--actions", nargs="+", default=["action/arm/pose/position_rela", "action/arm/pose/rotation_6d_rela"])
    parser.add_argument("--future-num", type=int, default=1)
    parser.add_argument("--prefetch-items", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--topk", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = McapLeRobotDataset(
        McapLeRobotDatasetConfig(
            data_root=args.data_root,
            states=args.states,
            images=args.images,
            actions=args.actions,
            horizon=HorizonConfig(fill_with_last=True, future_num=args.future_num),
            prefetch_items=args.prefetch_items,
        )
    )

    print(f"num_frames={dataset.num_frames} num_episodes={dataset.num_episodes}")
    profiler = cProfile.Profile()
    total_items = 0
    start = time.perf_counter()
    profiler.enable()
    for r in range(args.repeat):
        for i, _item in enumerate(dataset):
            print(f"[{r}] item {i} done")
            total_items += 1
    profiler.disable()
    elapsed = time.perf_counter() - start

    print(f"total_items={total_items}")
    print(f"elapsed_s={elapsed:.4f}")
    print(f"items_per_s={total_items / elapsed:.4f}")
    print(f"ms_per_item={elapsed / total_items * 1000:.4f}")

    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumtime").print_stats(args.topk)
    print(output.getvalue())


if __name__ == "__main__":
    main()
