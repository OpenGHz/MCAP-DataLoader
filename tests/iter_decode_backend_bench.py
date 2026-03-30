#!/home/ghz/.mini_conda3/envs/lerobot/bin/python3

from __future__ import annotations

import argparse
from statistics import mean
from time import perf_counter
from pathlib import Path
from mcap_data_loader.utils.av_coder import AvCoder, AvCoderConfig, DecodeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AvCoder.iter_decode() with pyav and torchcodec backends."
    )
    parser.add_argument("video_path", type=Path, help="Path to a local video file.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs for each backend.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit. Use 0 to decode the whole video.",
    )
    parser.add_argument(
        "--frame-format",
        choices=["bgr24", "rgb24"],
        default="rgb24",
        help="Frame format passed into iter_decode().",
    )
    parser.add_argument(
        "--target-time-base",
        type=int,
        default=int(1e9),
        help="Timestamp time base passed into iter_decode(). Use 0 to disable timestamps.",
    )
    return parser.parse_args()


def consume_iter_decode(
    video_path: Path,
    backend: str,
    frame_format: str,
    target_time_base: int,
    max_frames: int,
) -> tuple[int, int, int]:
    coder = AvCoder(AvCoderConfig())
    config = DecodeConfig(
        backend=backend,
        frame_format=frame_format,
        ensure_base_stamp=False,
        target_time_base=target_time_base,
    )

    frame_count = 0
    checksum = 0
    last_timestamp = -1

    for item in coder.iter_decode(str(video_path), config):
        if isinstance(item, dict):
            frame = item["data"]
            timestamp = item["t"]
            last_timestamp = int(timestamp)
        else:
            frame = item

        # Touch a tiny portion of decoded data so the benchmark measures actual consumption.
        checksum += int(frame.reshape(-1)[0])
        frame_count += 1

        if max_frames > 0 and frame_count >= max_frames:
            break

    coder.close()
    return frame_count, checksum, last_timestamp


def benchmark_backend(
    video_path: Path,
    backend: str,
    repeats: int,
    frame_format: str,
    target_time_base: int,
    max_frames: int,
) -> tuple[list[float], tuple[int, int, int]]:
    durations = []
    summary = None
    for _ in range(repeats):
        start = perf_counter()
        summary = consume_iter_decode(
            video_path,
            backend,
            frame_format,
            target_time_base,
            max_frames,
        )
        durations.append(perf_counter() - start)
    assert summary is not None
    return durations, summary


def print_summary(name: str, durations: list[float], summary: tuple[int, int, int]) -> None:
    frame_count, checksum, last_timestamp = summary
    print(
        f"{name}: "
        f"min={min(durations):.6f}s "
        f"avg={mean(durations):.6f}s "
        f"runs={len(durations)} "
        f"frames={frame_count} "
        f"checksum={checksum} "
        f"last_timestamp={last_timestamp}"
    )


def main() -> int:
    args = parse_args()

    pyav_durations, pyav_summary = benchmark_backend(
        args.video_path,
        "pyav",
        args.repeats,
        args.frame_format,
        args.target_time_base,
        args.max_frames,
    )
    print_summary("pyav", pyav_durations, pyav_summary)

    torchcodec_durations, torchcodec_summary = benchmark_backend(
        args.video_path,
        "torchcodec",
        args.repeats,
        args.frame_format,
        args.target_time_base,
        args.max_frames,
    )

    print_summary("torchcodec", torchcodec_durations, torchcodec_summary)

    pyav_min = min(pyav_durations)
    torchcodec_min = min(torchcodec_durations)
    if torchcodec_min > 0:
        print(f"torchcodec_speedup_vs_pyav={pyav_min / torchcodec_min:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
