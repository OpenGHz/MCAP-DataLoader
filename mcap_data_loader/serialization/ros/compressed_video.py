import io
import av
import numpy as np
from foxglove_msgs.msg import CompressedVideo
from mcap_data_loader.serialization.video.pyav import AvCoder, AvCoderConfig, AvCoderBasicConfig
from mcap_data_loader.serialization.ros import time_ns_to_stamp, stamp_from_dict
from typing import Dict
from pydantic import field_validator


class CompressedVideoEncoderConfig(AvCoderBasicConfig):
    """Configuration for CompressedVideoEncoder."""

    frame_id: str = "camera_color_optical_frame"
    """Frame ID to set in the CompressedVideo messages. Default is "camera_color_optical_frame"."""
    gop_size: int = 10
    """Group of Pictures (GOP) size for inter-frame compression. Default is 10, meaning one keyframe followed by 9 delta frames."""
    codec_options: Dict[str, str] = {"preset": "ultrafast", "tune": "zerolatency"}
    """Options to pass to the underlying H.264 encoder. Defaults to ultrafast preset and zerolatency tune for low-latency streaming."""

    @field_validator("codec_options", mode="after")
    def validate_codec_options(cls, v: dict):
        tune = v.get("tune")
        if tune != "zerolatency":
            if tune is not None:
                print(
                    f"Warning: Invalid tune option '{tune}'. Using 'zerolatency' instead."
                )
            v["tune"] = "zerolatency"
        return v


class CompressedVideoEncoder:
    """Stateful H.264 encoder that leverages inter-frame compression (P-frames).

    Built on top of :class:`AvCoder` in raw codec context mode (no container),
    so each :meth:`encode` call returns Annex B packets for exactly one frame
    (keyframe or delta frame), as required by the ``foxglove_msgs`` spec.
    """

    def __init__(self, config: CompressedVideoEncoderConfig):
        self._config = config
        self._frame_id = config.frame_id
        self._gop_size = config.gop_size
        av_config = AvCoderConfig(
            time_base=config.time_base,
            frame_format=config.frame_format,
            container_format=None,
            codec_options=config.codec_options,
            blocking=True,
        )
        self._coder = AvCoder(av_config)
        self._frame_index = 0
        self._configured = False

    def reset(self):
        """Reset encoder state for a new encoding session."""
        self._coder.reset()
        self._frame_index = 0

    def configure_stream(
        self,
        width: int,
        height: int,
        gop_size: int | None = None,
        max_b_frames: int = 0,
    ):
        """Configure the encoder stream parameters."""
        self._coder.configure_stream(
            width,
            height,
            gop_size=gop_size if gop_size is not None else self._gop_size,
            max_b_frames=max_b_frames,
        )

    def _encode_frame(self, image_rgb: np.ndarray) -> bytes:
        timestamp = self._frame_index
        if not self._configured:
            self.configure_stream(image_rgb.shape[1], image_rgb.shape[0])
            self._configured = True
        packets = self._coder.encode_frame_blocking(image_rgb, timestamp)
        data = b"".join(bytes(p) for p in packets)
        self._frame_index += 1
        return data

    def encode(
        self,
        image_rgb: np.ndarray,
        *,
        timestamp_sec: float | None = None,
    ) -> CompressedVideo:
        """Encode one RGB frame and return a ``CompressedVideo`` message."""
        if timestamp_sec is None:
            timestamp_sec = self._frame_index / self._config.time_base
        data = self._encode_frame(image_rgb)
        msg = CompressedVideo()
        msg.timestamp = time_ns_to_stamp(int(timestamp_sec * 1e9))
        msg.frame_id = self._frame_id
        msg.format = "h264"
        msg.data = data
        return msg

    def encode_image_dict(self, image_dict: dict):
        """Encode an image dict with keys 'data' (RGB uint8 array) and optional 'timestamp_sec'."""
        image_rgb = image_dict["data"]
        data = self._encode_frame(image_rgb)
        header = image_dict["header"]
        msg = CompressedVideo()
        msg.timestamp = stamp_from_dict(header["stamp"])
        msg.frame_id = header.get("frame_id", self._frame_id)
        msg.format = "h264"
        msg.data = data
        return msg

    def end(self, reset: bool = True):
        return self._coder.end(reset=reset)

    def close(self) -> None:
        self._coder.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def encode_compressed_video(
    image_rgb: np.ndarray,
    *,
    frame_id: str = "camera_color_optical_frame",
    fps: int = 30,
    timestamp_sec: float = 0.0,
) -> CompressedVideo:
    """Encode a single RGB frame (as an I-frame) into a CompressedVideo message."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")

    height, width = image_rgb.shape[:2]
    with CompressedVideoEncoder(
        width, height, fps=fps, gop_size=1, frame_id=frame_id
    ) as encoder:
        return encoder.encode(image_rgb, timestamp_sec=timestamp_sec)


def decode_compressed_video_to_rgb(msg: CompressedVideo) -> np.ndarray:
    """Decode a single-frame CompressedVideo message via a temporary bitstream container."""
    if msg.format != "h264":
        raise ValueError(f"Unsupported CompressedVideo format: {msg.format}")

    container = av.open(io.BytesIO(bytes(msg.data)), mode="r", format="h264")
    frames = list(container.decode(video=0))
    container.close()
    if len(frames) != 1:
        raise ValueError(
            f"Expected exactly one decoded frame, but received {len(frames)} frames"
        )
    return frames[0].to_ndarray(format="rgb24")


def decode_compressed_video(
    msg: CompressedVideo,
    decoder: av.CodecContext | None = None,
) -> np.ndarray:
    """Decode CompressedVideo with the lower-level Packet API used in stream readers."""
    if msg.format != "h264":
        raise ValueError(f"Unsupported CompressedVideo format: {msg.format}")

    local_decoder = decoder or av.CodecContext.create("h264", "r")
    frames = local_decoder.decode(av.Packet(bytes(msg.data)))
    if len(frames) != 1:
        raise ValueError(
            f"Expected exactly one decoded frame, but received {len(frames)} frames"
        )
    return frames[0].to_ndarray(format="rgb24")


def decode_compressed_video_sequence_to_rgb(
    messages: list[CompressedVideo],
) -> list[np.ndarray]:
    """Decode a sequence of CompressedVideo messages with one shared decoder."""
    decoder = av.CodecContext.create("h264", "r")
    decoded_images = []

    for msg in messages:
        if msg.format != "h264":
            raise ValueError(f"Unsupported CompressedVideo format: {msg.format}")
        frames = decoder.decode(av.Packet(bytes(msg.data)))
        if len(frames) != 1:
            raise ValueError(
                f"Expected exactly one decoded frame, but received {len(frames)} frames"
            )
        decoded_images.append(frames[0].to_ndarray(format="rgb24"))

    remaining_frames = decoder.decode()
    if remaining_frames:
        raise ValueError(
            f"Expected no buffered frames after flush, but received {len(remaining_frames)}"
        )
    return decoded_images


def load_rgb_frames_from_video(
    video_path: str,
    *,
    max_frames: int = 8,
) -> tuple[list[np.ndarray], float]:
    container = av.open(video_path)
    stream = container.streams.video[0]
    average_rate = float(stream.average_rate) if stream.average_rate else 30.0

    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= max_frames:
            break

    container.close()
    if not frames:
        raise ValueError(f"No video frames found in {video_path}")
    return frames, average_rate


def encode_video_frames_to_messages(
    frames: list[np.ndarray],
    *,
    fps: float,
    gop_size: int = 10,
    frame_id: str = "camera_color_optical_frame",
) -> list[CompressedVideo]:
    if not frames:
        return []
    height, width = frames[0].shape[:2]
    int_fps = max(1, round(fps))
    messages = []
    with CompressedVideoEncoder(
        width, height, fps=int_fps, gop_size=gop_size, frame_id=frame_id
    ) as encoder:
        for index, image_rgb in enumerate(frames):
            messages.append(encoder.encode(image_rgb, timestamp_sec=index / fps))
    return messages


def demo_compressed_video_codec() -> None:
    VIDEO_PATH = "/home/ghz/视频/window-new.mp4"
    source_frames, fps = load_rgb_frames_from_video(VIDEO_PATH, max_frames=8)
    sequence_msgs = encode_video_frames_to_messages(source_frames, fps=fps)

    msg = sequence_msgs[0]
    decoded = decode_compressed_video_to_rgb(msg)
    decoded_from_packet = decode_compressed_video(msg)
    decoded_sequence = decode_compressed_video_sequence_to_rgb(sequence_msgs)

    first_frame_error = np.abs(
        decoded.astype(np.int16) - source_frames[0].astype(np.int16)
    ).mean()
    print("CompressedVideo format:", msg.format)
    print("CompressedVideo bytes:", len(msg.data))
    print("Decoded image shape:", decoded.shape)
    print("Video source:", VIDEO_PATH)
    print("Video fps:", fps)
    print("Encoded messages:", len(sequence_msgs))
    print("First frame mean abs error:", first_frame_error)
    print(
        "Container-vs-packet mean diff:",
        np.abs(decoded.astype(np.int16) - decoded_from_packet.astype(np.int16)).mean(),
    )
    print("Sequence decoded frames:", len(decoded_sequence))

    assert msg.format == "h264"
    assert bytes(msg.data).startswith(b"\x00\x00\x00\x01")
    assert decoded.shape == source_frames[0].shape
    assert decoded_from_packet.shape == source_frames[0].shape
    assert np.array_equal(decoded, decoded_from_packet)
    assert len(decoded_sequence) == len(source_frames)
    for decoded_image, source_image in zip(decoded_sequence, source_frames):
        assert decoded_image.shape == source_image.shape
        assert (
            np.abs(
                decoded_image.astype(np.int16) - source_image.astype(np.int16)
            ).mean()
            < 8
        )
    # H.264 is lossy, so we only check that reconstruction is close enough.
    assert first_frame_error < 8


if __name__ == "__main__":
    demo_compressed_video_codec()
