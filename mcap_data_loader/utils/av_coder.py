import av
import numpy as np
import fractions
import json
from io import BytesIO
from typing import List, Optional, Union, Literal, Dict
from collections.abc import Generator
from turbojpeg import TurboJPEG
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from mcap_data_loader.basis import DataStamped, StrEnum
from mcap_data_loader.basis.cfgable import InitConfigMixin
from pydantic import BaseModel, PositiveInt, NonNegativeInt, ConfigDict
from enum import auto
from time import time_ns
from pathlib import Path


class DecodeConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    thread_type: str = "AUTO"
    """Threading type for decoding. `AUTO` lets PyAV decide."""
    frame_format: str = "bgr24"
    """Format of the frames to decode."""
    mismatch_tolerance: NonNegativeInt = 0
    """Number of frames that can be missing before raising an error."""
    ensure_base_stamp: bool = False
    """If True, ensures that the base timestamp is present in the video metadata."""
    target_time_base: PositiveInt = int(1e9)
    """Time base for the timestamps. If set to 0, timestamps are not returned."""


class NonMonotonicTimeMode(StrEnum):
    ADJUST = auto()
    """Adjust the timestamp to be just greater than the last timestamp."""
    DROP = auto()
    """Drop the frame with non-monotonic timestamp."""
    RAISE = auto()
    """Raise an error when a non-monotonic timestamp is encountered."""
    NONE = auto()
    """Do nothing. May result in out-of-order frames."""


class AvCoderConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    time_base: PositiveInt = int(1e6)
    """Time base for the encoder/decoder. Default is 1e6 (microseconds).
    Large time base (e.g. 1e9) improves timestamp precision but may cause overflow issues in some machines."""
    frame_format: str = "bgr24"
    """Format of the frames to encode/decode."""
    blocking: bool = True
    """Whether to use blocking encoding. If False, uses a separate thread."""
    log_level: Optional[int] = None
    """Logging level for the PyAV module."""
    non_monotonic_mode: NonMonotonicTimeMode = NonMonotonicTimeMode.ADJUST
    """Mode to handle frames with the same timestamp."""
    non_monotonic_log: bool = True
    """Whether to log when frames have the same timestamp."""


PathLike = Union[str, Path]


class AvCoder(InitConfigMixin):
    """
    A class for encoding video frames using PyAV.
    This class supports encoding frames in various formats and ensures that
    timestamps are strictly increasing.
    It can handle both NumPy arrays and raw byte data for frames.
    """

    logging = av.logging

    def __init__(self, config: AvCoderConfig):
        self.config = config
        av.logging.set_level(config.log_level)
        self._time_base = fractions.Fraction(1, config.time_base)
        self._configured = False
        self._frame_format = config.frame_format
        self._preprocess = None
        self._last_future = None
        self._outbuf = None
        self._container = None
        # NOTE: set max_workers to 1 to ensure frames are processed in order
        self._executor = None if config.blocking else ThreadPoolExecutor(1, "av_coder")
        self._ns2base = int(1e9 / self.config.time_base)
        self._encode_lock = Lock()
        self.reset()

    def reset(self, file_path: PathLike = ""):
        """
        Reset the encoder state.
        This method clears the output buffer and resets the start and last timestamps.
        Args:
            file_path (PathLike): Optional file path to save the encoded video for the following encoding session.
        """
        if self._last_future:
            self._last_future.result()
        self._close()
        self.set_output(file_path)
        self._start_time = 0
        self._last_time = 0
        self._configured = False
        self._last_future = None
        self._perf_logs = {}

    def set_output(self, file_path: PathLike):
        """
        Set the output file path for the encoder.
        This method closes the current container and opens a new one with the specified file path.
        Args:
            file_path (PathLike): The file path to save the encoded video.
        """
        self._outbuf = None if file_path else BytesIO()
        self._container = av.open(file_path or self._outbuf, "w", format="mp4")
        self.stream = self._container.add_stream("h264", options={"preset": "fast"})
        self.stream.codec_context.time_base = self._time_base
        self.stream.time_base = self._time_base

    def configure_stream(
        self,
        width: int,
        height: int,
        pix_fmt: Literal["yuv420p", "rgb24"] = "yuv420p",
    ):
        """
        Configure the stream with the given parameters.
        """
        stream = self.stream
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt
        self._configured = True

    def set_frame_type(self, frame_type: str):
        if frame_type == "bytes":
            jpeg = TurboJPEG()
            self._preprocess = jpeg.decode
        elif frame_type == "ndarray":
            self._preprocess = lambda x: x
        else:
            raise ValueError(f"Unsupported frame type: {frame_type}")

    def _set_frame_type(self, frame: Union[np.ndarray, bytes]):
        """
        Set the frame type based on the input frame.
        This method is called internally to determine how to process the frame.
        """
        if isinstance(frame, bytes):
            self.set_frame_type("bytes")
        elif isinstance(frame, np.ndarray):
            self.set_frame_type("ndarray")
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

    def _encode_frame(
        self,
        frame: Union[np.ndarray, bytes],
        timestamp: int,
        ns_to_base: bool = False,
    ):
        """
        Encode a single video frame with the given timestamp.
        Args:
            frame (Union[np.ndarray, bytes]): The video frame to encode.
            timestamp (int): The timestamp for the frame in the time base or nanoseconds (if ns_to_base is True).
            ns_to_base (bool): Whether to convert the timestamp from nanoseconds to the time base.
        """
        # start = time.monotonic()
        assert isinstance(timestamp, int), "Timestamp must be an integer"
        timestamp = timestamp // self._ns2base if ns_to_base else timestamp
        if self._start_time == 0:
            assert timestamp > 0, "Timestamp must be greater than 0"
            self._start_time = timestamp
            self._container.metadata["comment"] = json.dumps({"base_stamp": timestamp})
        if self._preprocess is None:
            self._set_frame_type(frame)
        video_frame = av.VideoFrame.from_ndarray(
            self._preprocess(frame), format=self._frame_format
        )
        if not self._configured:
            self.configure_stream(video_frame.width, video_frame.height)
        # Ensure timestamps are strictly increasing
        last_time = self._last_time
        if timestamp <= last_time:
            mode = self.config.non_monotonic_mode
            error_msg = f"Frame timestamp {timestamp} is not greater than last timestamp {last_time}"
            if mode is NonMonotonicTimeMode.RAISE:
                raise ValueError(error_msg)
            else:
                if self.config.non_monotonic_log:
                    self.get_logger().warning(error_msg + f", {mode}")
                if mode is NonMonotonicTimeMode.DROP:
                    return
                elif mode is NonMonotonicTimeMode.ADJUST:
                    timestamp = last_time + max(self._time_base.denominator // 1000, 1)
        self._last_time = timestamp
        video_frame.pts = timestamp - self._start_time
        video_frame.time_base = self._time_base
        packets = self.stream.encode(video_frame)
        self._container.mux(packets)
        # self._perf_logs["encode"] =  time.monotonic() - start

    def encode_frame(
        self,
        frame: Union[np.ndarray, bytes],
        timestamp: Optional[int] = None,
        ns_to_base: bool = False,
    ):
        """
        Encode a video frame with the given timestamp.
        Args:
            frame (Union[np.ndarray, bytes]): The video frame to encode.
            timestamp (Optional[int]): The timestamp for the frame in nanoseconds.
                If None, the current time will be used.
        """
        with self._encode_lock:
            timestamp = timestamp if timestamp is not None else time_ns()
            if self._executor is not None:
                self._last_future = self._executor.submit(
                    self._encode_frame, frame, timestamp, ns_to_base
                )
            else:
                self._encode_frame(frame, timestamp, ns_to_base)

    def end(self, file_path: PathLike = "", reset: bool = True) -> Optional[bytes]:
        """
        Finalize the encoding process.
        Args:
            file_path (PathLike): Optional file path to save the encoded video.
        Returns:
            Optional[bytes]: The encoded video bytes if no file is given.
        """
        with self._encode_lock:
            if self._last_future:
                self._last_future.result()
            packets = self.stream.encode()
            self._container.mux(packets)
            self._container.close()  # must close before getting value
            self._container = None
            value = None
            if self._outbuf is not None:
                value = self._outbuf.getvalue()
                if file_path:
                    with open(file_path, "wb") as f:
                        f.write(value)
            self._close()
            if reset:
                self.reset()
            return value

    def _close(self):
        if self._container is not None:
            self._container.close()
            self._container = None
        if self._outbuf is not None:
            self._outbuf.close()
            self._outbuf = None

    def close(self):
        """
        Close the encoder and release resources.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._close()

    @classmethod
    def get_logger(cls):
        """
        Returns a logger instance for logging purposes.
        """
        return getLogger(cls.__name__)

    @classmethod
    def _init_decode(
        cls,
        video: Union[str, bytes],
        thread_type: str = "AUTO",
        ensure_base_stamp: bool = False,
    ):
        if isinstance(video, bytes):
            container = av.open(BytesIO(video))
        else:
            container = av.open(video, "r")
        # Enable multithreading for decoding
        video_stream = container.streams.video[0]
        video_stream.thread_type = thread_type
        meta_comment = container.metadata.get("comment", "{}")
        try:
            comment: dict = json.loads(meta_comment)
        except json.JSONDecodeError:
            meta_comment = meta_comment.replace("'", '"')
            comment: dict = json.loads(meta_comment)

        base_stamp = comment.get("base_stamp", None)
        if base_stamp is None:
            assert not ensure_base_stamp, (
                "Base timestamp not found in video metadata. "
                "Set ensure_base_stamp to True to raise an error."
            )
            cls.get_logger().warning(
                "Base timestamp not found in video metadata. Using 0 as base."
            )
            base_stamp = 0
        else:
            if not isinstance(base_stamp, int):
                cls.get_logger().warning(
                    f"Base timestamp is not an integer: {base_stamp}. "
                    "Converting to integer."
                )
                base_stamp = int(base_stamp)
        return container, video_stream, base_stamp, video_stream.frames

    @classmethod
    def decode(
        cls,
        video: Union[str, bytes],
        indices: Optional[List[int]] = None,
        frame_format: str = "bgr24",
        thread_type: str = "AUTO",
        mismatch_tolerance: int = 0,
        ensure_base_stamp: bool = True,
    ) -> Union[List[np.ndarray], Dict[int, np.ndarray]]:
        """
        Reads all frames from a video file using PyAV.
        Args:
            video_path (str): Path to the video file or the encoded video bytes.
        Returns:
            List[np.ndarray]: A list of frames, each represented as a NumPy array.
        """
        container, video_stream, base_stamp, frame_cnt = cls._init_decode(
            video, thread_type, ensure_base_stamp=ensure_base_stamp
        )
        if indices is not None:
            indices = sorted(set(indices))
            end_index = indices[-1]
            assert 0 <= end_index < frame_cnt, f"{end_index} out of bounds"
            frames = {}
            exp_cnt = len(indices)
        else:
            frames = []
            exp_cnt = frame_cnt
        for index, frame in enumerate(container.decode(video=0)):
            frame_arr = frame.to_ndarray(format=frame_format)
            if indices is None:
                frames.append(frame_arr)
            elif index == indices[0]:
                frames[index] = frame_arr
                indices.pop(0)
                if not indices:
                    break
        if mismatch_tolerance:
            if indices is None:
                missing_cnt = frame_cnt - len(frames)
                if missing_cnt > 0 and missing_cnt <= mismatch_tolerance:
                    cls.get_logger().warning(
                        f"Missing {missing_cnt} frames in video. Filling with last frame."
                    )
                    for _ in range(missing_cnt):
                        frames.append(frame_arr)
            elif indices:
                if len(indices) <= mismatch_tolerance:
                    cls.get_logger().warning(
                        f"Frame indices {indices} not found in video. Filling with last frame."
                    )
                    for index in indices:
                        frames[index] = frame_arr
        # do not close since it will block the code
        # container.close()
        assert len(frames) == exp_cnt, (
            f"Frame count mismatch: {len(frames)} != {exp_cnt}; indices: {indices} frame_cnt: {frame_cnt}"
        )
        return frames

    @classmethod
    def iter_decode(
        cls,
        video: Union[bytes, str],
        config: Optional[DecodeConfig] = None,
    ) -> Generator[Union[DataStamped[np.ndarray], np.ndarray]]:
        """
        Generator to decode frames from a video file. This method yields frames one by one.
        Args:
            video (Union[bytes, str]): The video file path or the encoded video bytes.
            thread_type (str): The threading type for decoding. Defaults to "AUTO".
            frame_format (str): The format of the frames to decode. Defaults to "bgr24".
            mismatch_tolerance (int): The number of frames that can be missing before raising an error.
                Defaults to 0, which means no tolerance.
            ensure_base_stamp (bool): If True, ensures that the base timestamp is present in the video metadata.
                If not present, raises an error. Defaults to False.
            target_time_base (int): The time base for the timestamps. Defaults to 1e9 (nanoseconds).
        Yields:
            Union[tuple[np.ndarray, int], np.ndarray]: A tuple of the frame and its absolute timestamp
                if target_time_base > 0, otherwise just the frame.
        """
        config = config or DecodeConfig()
        container, video_stream, base_stamp, frame_cnt = cls._init_decode(
            video, config.thread_type, config.ensure_base_stamp
        )
        # cls.get_logger().info(f"Decoding video with {frame_cnt} frames.")
        cnt = 0
        time_factor = (
            fractions.Fraction(config.target_time_base, 1) * video_stream.time_base
        )
        for frame in container.decode(video=0):
            cnt += 1
            np_frame = frame.to_ndarray(format=config.frame_format)
            if config.target_time_base:
                abs_stamp = int((base_stamp + frame.pts) * time_factor)
                yield {"data": np_frame, "t": abs_stamp}
            else:
                yield np_frame
        mismatch = frame_cnt - cnt
        if mismatch > 0:
            if mismatch <= config.mismatch_tolerance:
                cls.get_logger().warning(
                    f"Missing {mismatch} frames in video. Filling with last frame."
                )
                for _ in range(mismatch):
                    if config.target_time_base:
                        yield {"data": np_frame, "t": abs_stamp}
                    else:
                        yield np_frame
            else:
                raise ValueError(
                    f"Frame count mismatch: {cnt} != {frame_cnt}; "
                    f"mismatch tolerance: {config.mismatch_tolerance}"
                )
        elif mismatch < 0:
            raise ValueError(
                f"Frame count mismatch: {cnt} != {frame_cnt}; "
                f"mismatch tolerance: {config.mismatch_tolerance}"
            )

    @classmethod
    def seek_frames(
        cls,
        video: Union[bytes, str],
        start_time: int,
        interval: int = 0,
        step: int = 0,
        end_time: Optional[int] = None,
        thread_type: str = "AUTO",
        frame_format: str = "bgr24",
        ensure_base_stamp: bool = False,
    ) -> List[np.ndarray]:
        # TODO: implement according to the test_av_seek.py
        container, video_stream, base_stamp, frame_cnt = cls._init_decode(
            video, thread_type, ensure_base_stamp
        )
        container.seek(start_time, stream=video_stream, backward=True, any_frame=False)
        last_frame = None
        frame_cnt = 0
        target_frames = []
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                frame_cnt += 1
                if frame.pts >= start_time:
                    if last_frame is not None:
                        # print("Last frame pts:", last_frame.pts)
                        last_delta = start_time - last_frame.pts
                        current_delta = frame.pts - start_time
                        if last_delta < current_delta:
                            target_frame = last_frame
                            # print("Found frame before target timestamp")
                        else:
                            target_frame = frame
                            # print("Found frame after target timestamp")
                    else:
                        target_frame = frame
                        # print("Found first frame after target timestamp")
                    break
                last_frame = frame
            else:
                continue
            break
        else:
            target_frame = None
            cls.get_logger().warning(
                "No frame found after seeking to target timestamp. The last frame pts is:",
                frame.pts,
            )
        cls.get_logger().info("Total frames processed:", frame_cnt)
