#!/home/ghz/.mini_conda3/envs/lerobot/bin/python3
"""Decode a video with AvCoder (PyAV) and re-encode it with NvcCoder (NVENC).

Example:
    python tests/nvc_reencode.py input.mp4 output.mp4 --fps 30 --bitrate 4000000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from mcap_data_loader.serialization.video.pyav import (
    AvCoder,
    AvCoderConfig,
    DecodeConfig,
)
from mcap_data_loader.serialization.video.nvc import NvcCoder, NvcCoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input video path.")
    parser.add_argument("output", type=Path, help="Output video path (.mp4).")
    parser.add_argument("--fps", type=int, default=30, help="Encoder fps hint.")
    parser.add_argument(
        "--bitrate", type=int, default=4_000_000, help="Target bitrate (bps)."
    )
    parser.add_argument(
        "--preset",
        default="P4",
        help="NVENC preset (P1=fastest .. P7=highest quality).",
    )
    parser.add_argument(
        "--frame-format",
        choices=["bgr24", "rgb24"],
        default="rgb24",
        help="Pixel layout shared by decoder and encoder.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit. Use 0 to process the whole video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    decoder = AvCoder(AvCoderConfig())
    decode_cfg = DecodeConfig(
        frame_format=args.frame_format,
        target_time_base=int(1e9),
        ensure_base_stamp=False,
    )

    encoder = NvcCoder(
        NvcCoderConfig(
            frame_format=args.frame_format,
            container_format=args.output.suffix.lstrip(".") or "mp4",
            fps=args.fps,
            bitrate=args.bitrate,
            preset=args.preset,
        )
    )
    encoder.reset(str(args.output))

    start = perf_counter()
    frame_count = 0
    for item in decoder.iter_decode(str(args.input), decode_cfg):
        if isinstance(item, dict):
            frame, timestamp = item["data"], int(item["t"])
        else:
            frame, timestamp = item, frame_count * int(1e9 / args.fps)
        encoder.encode_frame(frame, timestamp=timestamp, ns_to_base=True)
        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            break

    encoder.end()
    decoder.close()
    encoder.close()

    elapsed = perf_counter() - start
    size = args.output.stat().st_size
    print(
        f"re-encoded {frame_count} frames in {elapsed:.2f}s "
        f"({frame_count / elapsed:.1f} fps) -> {args.output} ({size} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
