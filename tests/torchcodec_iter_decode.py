#!/home/ghz/.mini_conda3/envs/lerobot/bin/python3

from __future__ import annotations

from statistics import mean
from pathlib import Path
from time import perf_counter
from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder
import argparse
import av
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal TorchCodec example for sequential frame-by-frame decoding "
            "without depending on LeRobot."
        )
    )
    parser.add_argument("video_path", type=Path, help="Path to a local video file.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Maximum number of frames to print.",
    )
    parser.add_argument(
        "--dimension-order",
        choices=["NCHW", "NHWC"],
        default="NCHW",
        help="TorchCodec output layout.",
    )
    parser.add_argument(
        "--seek-mode",
        choices=["exact", "approximate"],
        default="exact",
        help="TorchCodec seek mode.",
    )
    parser.add_argument(
        "--num-ffmpeg-threads",
        type=int,
        default=1,
        help="Number of FFmpeg decoding threads.",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=64,
        help="Number of frames used for efficiency comparison.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=5,
        help="Number of repeated timing runs for each method.",
    )
    return parser.parse_args()


def decode_frames_one_by_one(decoder, frame_indices: list[int]):

    frames = []
    for frame_index in frame_indices:
        frames.append(decoder.get_frame_at(frame_index))

    return FrameBatch(
        data=torch.stack([frame.data for frame in frames], dim=0),
        pts_seconds=torch.tensor([frame.pts_seconds for frame in frames]),
        duration_seconds=torch.tensor([frame.duration_seconds for frame in frames]),
    )


def decode_frames_in_batch(decoder, frame_indices: list[int]):
    return decoder.get_frames_at(frame_indices)


def decode_frames_in_range(decoder, frame_indices: list[int]):
    if not frame_indices:
        raise ValueError("frame_indices must not be empty.")
    start = frame_indices[0]
    stop = frame_indices[-1] + 1
    return decoder.get_frames_in_range(start, stop)


def decode_frames_with_pyav(
    video_path: Path,
    frame_indices: list[int],
    dimension_order: str,
):
    if not frame_indices:
        raise ValueError("frame_indices must not be empty.")

    target_indices = set(frame_indices)
    last_index = frame_indices[-1]
    data_by_index: dict[int, torch.Tensor] = {}
    pts_by_index: dict[int, float] = {}
    duration_by_index: dict[int, float] = {}

    # hwaccel = av.codec.hwaccel.HWAccel(
    #     device_type="cuda",
    #     allow_software_fallback=False,
    # )
    hwaccel = None
    with av.open(str(video_path), hwaccel=hwaccel) as container:
        stream = container.streams.video[0]
        for decoded_index, frame in enumerate(container.decode(stream)):
            if decoded_index > last_index and len(data_by_index) == len(target_indices):
                break
            if decoded_index not in target_indices:
                continue

            array = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(array)
            if dimension_order == "NCHW":
                tensor = tensor.permute(2, 0, 1)
            data_by_index[decoded_index] = tensor

            if frame.time is None:
                pts_seconds = 0.0
            else:
                pts_seconds = float(frame.time)
            pts_by_index[decoded_index] = pts_seconds

            if frame.duration is None or frame.time_base is None:
                duration_seconds = 0.0
            else:
                duration_seconds = float(frame.duration * frame.time_base)
            duration_by_index[decoded_index] = duration_seconds

            if len(data_by_index) == len(target_indices):
                break

    missing = [index for index in frame_indices if index not in data_by_index]
    if missing:
        raise IndexError(f"PyAV failed to decode frame indices: {missing}")

    return {
        "data": torch.stack([data_by_index[index] for index in frame_indices], dim=0),
        "pts_seconds": torch.tensor([pts_by_index[index] for index in frame_indices]),
        "duration_seconds": torch.tensor(
            [duration_by_index[index] for index in frame_indices]
        ),
    }


def benchmark(name: str, fn, repeats: int) -> list[float]:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        durations.append(perf_counter() - start)
        if result is None:
            raise RuntimeError(f"{name} returned None unexpectedly.")
    return durations


def print_timing_summary(name: str, durations: list[float]) -> None:
    print(
        f"{name}: "
        f"min={min(durations):.6f}s "
        f"avg={mean(durations):.6f}s "
        f"runs={len(durations)}"
    )


def main() -> int:
    args = parse_args()

    def make_decoder():
        return VideoDecoder(
            args.video_path,
            dimension_order=args.dimension_order,
            seek_mode=args.seek_mode,
            num_ffmpeg_threads=args.num_ffmpeg_threads,
        )

    decoder = make_decoder()

    print("TorchCodec does not expose a public Python frame iterator on VideoDecoder.")
    print("Sequential decoding is typically done by iterating over frame indices.")
    print("For contiguous frames, get_frames_in_range() is the more direct batch API.")
    print("PyAV baseline below also materializes decoded frames into a stacked tensor.")
    print(f"video_path={args.video_path}")
    print(f"num_frames={len(decoder)}")
    print(f"dimension_order={args.dimension_order}")
    print(f"seek_mode={args.seek_mode}")

    limit = min(args.max_frames, len(decoder))
    for frame_index in range(limit):
        frame = decoder.get_frame_at(frame_index)
        print(
            f"frame_index={frame_index} "
            f"shape={tuple(frame.data.shape)} "
            f"dtype={frame.data.dtype} "
            f"pts_seconds={frame.pts_seconds:.6f} "
            f"duration_seconds={frame.duration_seconds:.6f}"
        )

    benchmark_count = min(args.benchmark_frames, len(decoder))
    frame_indices = list(range(benchmark_count))
    print(f"benchmark_frames={benchmark_count}")
    print(f"benchmark_repeats={args.benchmark_repeats}")

    if benchmark_count == 0:
        print("No frames available for benchmarking.")
        return 0

    # Warm up both paths once to reduce one-time initialization effects.
    decode_frames_one_by_one(make_decoder(), frame_indices[:1])
    decode_frames_in_batch(make_decoder(), frame_indices[:1])
    decode_frames_in_range(make_decoder(), frame_indices[:1])
    decode_frames_with_pyav(args.video_path, frame_indices[:1], args.dimension_order)

    single_durations = benchmark(
        "get_frame_at loop",
        lambda: decode_frames_one_by_one(make_decoder(), frame_indices),
        args.benchmark_repeats,
    )
    batch_durations = benchmark(
        "get_frames_at batch",
        lambda: decode_frames_in_batch(make_decoder(), frame_indices),
        args.benchmark_repeats,
    )
    range_durations = benchmark(
        "get_frames_in_range contiguous batch",
        lambda: decode_frames_in_range(make_decoder(), frame_indices),
        args.benchmark_repeats,
    )
    pyav_durations = benchmark(
        "pyav contiguous decode",
        lambda: decode_frames_with_pyav(
            args.video_path, frame_indices, args.dimension_order
        ),
        args.benchmark_repeats,
    )

    print_timing_summary("get_frame_at loop", single_durations)
    print_timing_summary("get_frames_at indexed batch", batch_durations)
    print_timing_summary("get_frames_in_range contiguous batch", range_durations)
    print_timing_summary("pyav contiguous decode", pyav_durations)

    batch_min = min(batch_durations)
    single_min = min(single_durations)
    if batch_min > 0:
        print(f"indexed_batch_speedup_vs_loop={single_min / batch_min:.2f}x")
    range_min = min(range_durations)
    if range_min > 0:
        print(f"contiguous_batch_speedup_vs_loop={single_min / range_min:.2f}x")
    pyav_min = min(pyav_durations)
    if pyav_min > 0:
        print(f"pyav_speedup_vs_loop={single_min / pyav_min:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
