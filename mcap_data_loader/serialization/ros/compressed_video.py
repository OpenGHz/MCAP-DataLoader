import io
import av
import genpy
import numpy as np
from foxglove_msgs.msg import CompressedVideo


def encode_compressed_video(
    image_rgb: np.ndarray,
    *,
    frame_id: str = "camera_color_optical_frame",
    fps: int = 30,
    timestamp_sec: float = 0.0,
) -> CompressedVideo:
    """Encode a single RGB frame into a foxglove_msgs/CompressedVideo message."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")

    height, width = image_rgb.shape[:2]
    buffer = io.BytesIO()

    # Foxglove's CompressedVideo currently expects H.264 Annex B payloads.
    container = av.open(buffer, mode="w", format="h264")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
    stream.codec_context.gop_size = 1
    stream.codec_context.max_b_frames = 0

    frame = av.VideoFrame.from_ndarray(image_rgb, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    msg = CompressedVideo()
    msg.timestamp = genpy.Time.from_sec(timestamp_sec)
    msg.frame_id = frame_id
    msg.format = "h264"
    msg.data = buffer.getvalue()
    return msg


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
    frame_id: str = "camera_color_optical_frame",
) -> list[CompressedVideo]:
    messages = []
    for index, image_rgb in enumerate(frames):
        messages.append(
            encode_compressed_video(
                image_rgb,
                frame_id=frame_id,
                fps=max(1, round(fps)),
                timestamp_sec=index / fps,
            )
        )
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
