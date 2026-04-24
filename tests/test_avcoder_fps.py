import numpy as np

from mcap_data_loader.serialization.video.pyav import (
    AvCoder,
    AvCoderConfig,
    DecodeConfig,
)


CODEC_OPTIONS = {"preset": "ultrafast", "tune": "zerolatency"}


def _roundtrip_timestamps(config: AvCoderConfig, timestamps_ns: list[int]) -> list[int]:
    coder = AvCoder(config)
    for index, timestamp_ns in enumerate(timestamps_ns):
        frame = np.full((16, 16, 3), index, dtype=np.uint8)
        coder.encode_frame(frame, timestamp=timestamp_ns, ns_to_base=True)
    video = coder.end(reset=False)
    coder.close()
    assert video is not None

    items = list(
        AvCoder.iter_decode(
            video,
            DecodeConfig(target_time_base=int(1e9), ensure_base_stamp=True),
        )
    )
    return [int(item["t"]) for item in items]


def test_avcoder_preserves_input_timestamps_by_default():
    timestamps_ns = [1_000_000_000, 1_130_000_000, 1_270_000_000]

    decoded_timestamps = _roundtrip_timestamps(
        AvCoderConfig(
            time_base=int(1e6),
            codec_options=CODEC_OPTIONS,
        ),
        timestamps_ns,
    )

    assert decoded_timestamps == timestamps_ns


def test_avcoder_can_encode_with_fixed_fps():
    timestamps_ns = [1_000_000_000, 1_130_000_000, 1_270_000_000]

    decoded_timestamps = _roundtrip_timestamps(
        AvCoderConfig(
            time_base=int(1e6),
            fps=10,
            codec_options=CODEC_OPTIONS,
        ),
        timestamps_ns,
    )

    assert decoded_timestamps == [1_000_000_000, 1_100_000_000, 1_200_000_000]
