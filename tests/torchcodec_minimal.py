#!/home/ghz/.mini_conda3/envs/lerobot/bin/python3

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.video_utils import decode_video_frames


@dataclass
class MinimalLeRobotConfig:
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="local/debug",
            root=".",
            video_backend="torchcodec",
        )
    )
    tolerance_s: float = 1 / 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal TorchCodec decoding example that follows LeRobot's config shape "
            "and calls lerobot.datasets.video_utils.decode_video_frames()."
        )
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to a local video file.",
    )
    parser.add_argument(
        "--timestamps",
        type=float,
        nargs="+",
        default=[0.0],
        help="Frame timestamps in seconds, matching LeRobot's decode_video_frames() usage.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1 / 30,
        help="Timestamp tolerance in seconds, matching cfg.tolerance_s in LeRobot.",
    )
    parser.add_argument(
        "--log-loaded-timestamps",
        action="store_true",
        help="Accepted for parity with LeRobot's torchcodec helper; this minimal example only decodes frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = MinimalLeRobotConfig()
    cfg.dataset.video_backend = "torchcodec"
    cfg.tolerance_s = args.tolerance_s

    frames = decode_video_frames(
        video_path=args.video_path,
        timestamps=args.timestamps,
        tolerance_s=cfg.tolerance_s,
        backend=cfg.dataset.video_backend,
    )

    print(f"video_backend={cfg.dataset.video_backend}")
    print(f"video_path={args.video_path}")
    print(f"timestamps={args.timestamps}")
    print(f"tolerance_s={cfg.tolerance_s}")
    print(f"frames.shape={tuple(frames.shape)}")
    print(f"frames.dtype={frames.dtype}")
    print(f"frames.min={frames.min().item():.6f}")
    print(f"frames.max={frames.max().item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
