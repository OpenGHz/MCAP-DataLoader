import numpy as np
import fractions
from typing import List, Optional, Union, Literal, Dict, Any, final
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from mcap_data_loader.basis import StrEnum
from mcap_data_loader.basis.cfgable import InitConfigABCMixin
from pydantic import BaseModel, PositiveInt, NonNegativeInt, ConfigDict
from enum import auto
from time import time_ns
from pathlib import Path
from abc import abstractmethod


Packet = Any


class DecodeConfig(BaseModel, frozen=True):
    """Configuration for video decoding."""

    model_config = ConfigDict(extra="forbid")

    frame_format: str = "bgr24"
    """Format of the frames to decode."""
    mismatch_tolerance: NonNegativeInt = 0
    """Number of frames that can be missing before raising an error."""
    ensure_base_stamp: bool = False
    """If True, ensures that the base timestamp is present in the video metadata."""
    target_time_base: PositiveInt = int(1e9)
    """Time base for the timestamps. If set to 0, timestamps are not returned."""
    dimension_order: Literal["NHWC", "NCHW"] = "NHWC"
    """Dimension order for torchcodec frames. PyAV always uses NHWC."""


class NonMonotonicTimeMode(StrEnum):
    """Mode to handle non-monotonic timestamps during encoding."""

    ADJUST = auto()
    """Adjust the timestamp to be just greater than the last timestamp."""
    DROP = auto()
    """Drop the frame with non-monotonic timestamp."""
    RAISE = auto()
    """Raise an error when a non-monotonic timestamp is encountered."""
    NONE = auto()
    """Do nothing. May result in out-of-order frames."""


class AvCoderBasicConfig(BaseModel, frozen=True):
    """Basic configuration for AvCoder."""

    model_config = ConfigDict(extra="forbid")

    time_base: PositiveInt = int(1e6)
    """Time base for the encoder/decoder. Default is 1e6 (microseconds).
    Large time base (e.g. 1e9) improves timestamp precision but may cause overflow issues in some machines."""
    fps: Optional[PositiveInt] = None
    """Optional fixed frame rate for encoding. When ``None``, input timestamps are preserved."""
    frame_format: str = "bgr24"
    """Format of the frames to encode/decode."""
    log_level: Optional[int] = None
    """Logging level for the PyAV module."""
    non_monotonic_mode: NonMonotonicTimeMode = NonMonotonicTimeMode.ADJUST
    """Mode to handle frames with the same timestamp."""
    non_monotonic_log: bool = True
    """Whether to log when frames have the same timestamp."""
    codec_options: Dict[str, str] = {"preset": "fast"}
    """Codec options passed to the encoder."""


class AvCoderConfig(AvCoderBasicConfig):
    """Configuration for AvCoder."""

    blocking: bool = True
    """Whether to use blocking encoding. If False, uses a separate thread."""
    container_format: Optional[str] = "mp4"
    """Container format for encoding. Set to None for raw codec context mode
    (outputs Annex B packets without a container, useful for per-frame packet extraction)."""


PathLike = Union[str, Path]


class AvCoderBasis(InitConfigABCMixin):
    """
    A class for encoding and decoding video frames using PyAV.
    This class supports encoding frames in various formats and ensures that
    timestamps are strictly increasing.
    It can handle both NumPy arrays and raw byte data for frames.
    """

    def __init__(self, config: AvCoderConfig):
        self.config = config
        self._set_log_level(config.log_level)
        self._time_base = fractions.Fraction(1, config.time_base)
        if config.fps is not None and config.fps > config.time_base:
            raise ValueError(
                f"fps ({config.fps}) must not exceed time_base ({config.time_base})"
            )
        self._fixed_frame_step = (
            fractions.Fraction(config.time_base, config.fps)
            if config.fps is not None
            else None
        )
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

    @abstractmethod
    def _set_log_level(self, level: int):
        """Set the logging level for the PyAV module."""

    @final
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
        self._start_time = None
        self._last_time = -1
        self._encoded_frame_count = 0
        self._configured = False
        self._last_future = None
        self._perf_logs = {}

    @final
    def _normalize_timestamp(self, timestamp: int, ns_to_base: bool) -> int:
        assert isinstance(timestamp, int), "Timestamp must be an integer"
        return timestamp // self._ns2base if ns_to_base else timestamp

    @final
    def _resolve_frame_timestamp(
        self, timestamp: int, ns_to_base: bool
    ) -> tuple[int, bool]:
        timestamp = self._normalize_timestamp(timestamp, ns_to_base)
        is_first_frame = self._start_time is None
        if is_first_frame:
            if timestamp < 0:
                raise ValueError("Timestamp must not be negative")
            self._start_time = timestamp
        if self._fixed_frame_step is None:
            return timestamp, is_first_frame

        frame_offset = self._fixed_frame_step * self._encoded_frame_count
        return (
            self._start_time + int(frame_offset + fractions.Fraction(1, 2)),
            is_first_frame,
        )

    @abstractmethod
    def set_output(self, file_path: PathLike = ""):
        """
        Set the output file path for the encoder.
        This method closes the current container and opens a new one with the specified file path.
        When ``container_format`` is ``None``, creates a raw codec context (no container).
        Args:
            file_path (PathLike): The file path to save the encoded video.
        """

    @abstractmethod
    def configure_stream(
        self,
        width: int,
        height: int,
        pix_fmt: Literal["yuv420p", "rgb24"] = "yuv420p",
        **codec_kwargs,
    ):
        """
        Configure the stream with the given parameters.
        Additional keyword arguments (e.g. ``gop_size``, ``max_b_frames``) are
        set as attributes on the stream / codec context.
        In raw codec context mode the context is opened after configuration.
        """

    @final
    def set_frame_type(self, frame_type: str):
        if frame_type == "bytes":
            from turbojpeg import TurboJPEG

            jpeg = TurboJPEG()
            self._preprocess = jpeg.decode
        elif frame_type == "ndarray":
            self._preprocess = lambda x: x
        else:
            raise ValueError(f"Unsupported frame type: {frame_type}")

    @final
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

    @abstractmethod
    def encode_frame_blocking(
        self,
        frame: Union[np.ndarray, bytes],
        timestamp: int,
        ns_to_base: bool = False,
    ) -> List[Packet]:
        """
        Encode a single video frame with the given timestamp.
        Args:
            frame (Union[np.ndarray, bytes]): The video frame to encode.
            timestamp (int): The timestamp for the frame in the time base or nanoseconds (if ns_to_base is True).
            ns_to_base (bool): Whether to convert the timestamp from nanoseconds to the time base.
        Returns:
            List[Packet]: A list of encoded packets for the frame.
        """

    @final
    def encode_frame(
        self,
        frame: Union[np.ndarray, bytes],
        timestamp: Optional[int] = None,
        ns_to_base: bool = False,
    ) -> Union[List[Packet], Future]:
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
                    self.encode_frame_blocking, frame, timestamp, ns_to_base
                )
                return self._last_future
            else:
                return self.encode_frame_blocking(frame, timestamp, ns_to_base)

    @final
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
            packets = self._end()
            value = None
            if self._outbuf is not None:
                value = self._outbuf.getvalue()
                if file_path:
                    with open(file_path, "wb") as f:
                        f.write(value)
            elif self.config.container_format is None:
                value = b"".join(bytes(p) for p in packets)
            self._close()
            if reset:
                self.reset()
            return value

    @abstractmethod
    def _end(self) -> List[Packet]:
        """Finalize the encoding process and return any remaining packets."""

    @abstractmethod
    def _close(self):
        """Close the encoder and release resources."""

    @final
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
