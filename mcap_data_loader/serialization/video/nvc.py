"""GPU-accelerated H.264 / HEVC encoder based on NVIDIA NVENC.

Exposes :class:`NvcCoder`, a drop-in sibling of :class:`AvCoder` that implements
:class:`AvCoderBasis` using PyNvVideoCodec. Raw Annex B output is produced on the
GPU; when ``container_format`` is set, the Annex B bitstream is remuxed through
PyAV to the requested container on finalize.

Limitations:
    - NVENC does not consume the per-frame timestamps passed to
      :meth:`encode_frame`. Container-mode timing is therefore reconstructed
      from ``fps`` by the PyAV muxer. Use raw mode and collect per-frame packet
      bytes if precise timestamps are required.
    - ``codec_options`` on :class:`AvCoderBasicConfig` is ignored; NVENC-specific
      knobs live on :class:`NvcCoderConfig`.
"""

import av
import ctypes
import gc
import numpy as np
import os
import torch
from io import BytesIO
from functools import lru_cache
from typing import List, Literal, Optional, Union

import PyNvVideoCodec as nvc
from pydantic import ConfigDict, NonNegativeInt, PositiveInt

from mcap_data_loader.serialization.video.basis import (
    AvCoderBasis,
    AvCoderConfig,
    NonMonotonicTimeMode,
    Packet,
    PathLike,
)
from mcap_data_loader.utils.nvc import (
    ensure_torch_dlpack_positional_stream_compat,
    rgb_to_nv12,
)


FrameInput = Union[np.ndarray, bytes, torch.Tensor]


class NvcCoderConfig(AvCoderConfig):
    """Configuration for :class:`NvcCoder`."""

    model_config = ConfigDict(extra="forbid")

    nvenc_codec: Literal["h264", "hevc"] = "h264"
    """NVENC codec name passed to ``nvc.CreateEncoder``."""
    fps: PositiveInt = 30
    """Frame rate hint for NVENC rate control (also used by the PyAV muxer)."""
    bitrate: PositiveInt = 4_000_000
    """Target average bitrate in bits per second."""
    preset: str = "P4"
    """NVENC preset (P1=fastest .. P7=slowest/highest quality)."""
    rc: str = "vbr"
    """Rate-control mode (``cbr`` / ``vbr`` / ``constqp`` ...)."""
    tuninginfo: str = "high_quality"
    """NVENC tuning info (``high_quality`` / ``low_latency`` / ``lossless`` ...)."""
    gop: Optional[PositiveInt] = None
    """GOP length. When ``None``, defaults to ``fps`` (1 IDR per second)."""
    bf: NonNegativeInt = 0
    """Number of consecutive B-frames between references. Default 0 keeps encode
    order equal to display order, which is required for the remux-to-container
    path (PyAV's raw h264 demuxer does not recover pts/dts reordering)."""


__all__ = ["NvcCoder", "NvcCoderConfig", "FrameInput"]


@lru_cache(maxsize=1)
def _get_cuda_driver():
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxGetCurrent.restype = ctypes.c_int
    cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    cuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxSetCurrent.restype = ctypes.c_int
    return cuda


def _check_cuda_result(status: int, func_name: str) -> None:
    if status != 0:
        raise RuntimeError(f"{func_name} failed with CUDA driver error code {status}")


@lru_cache(maxsize=None)
def _get_primary_cuda_context(device_id: int) -> int:
    cuda = _get_cuda_driver()
    ctx = ctypes.c_void_p()
    _check_cuda_result(
        cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device_id),
        "cuDevicePrimaryCtxRetain",
    )
    if not ctx.value:
        raise RuntimeError(
            f"Failed to retain primary CUDA context for device {device_id}"
        )
    return int(ctx.value)


def _format_cuda_visible_mapping(visible_device_id: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return f"visible_device={visible_device_id}, mapped_device={visible_device_id}"

    tokens = [token.strip() for token in visible_devices.split(",")]
    if 0 <= visible_device_id < len(tokens):
        mapped_device = tokens[visible_device_id]
    else:
        mapped_device = "out-of-range"
    return (
        f"visible_device={visible_device_id}, "
        f"mapped_device={mapped_device}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
    )


class NvcCoder(AvCoderBasis):
    """NVIDIA NVENC-backed video encoder.

    Accepted frame types:
        - ``np.ndarray`` – ``[H, W, 3]`` uint8 in ``frame_format`` order
          (``rgb24`` or ``bgr24``).
        - ``bytes`` – JPEG payload, decoded via TurboJPEG to an ndarray.
        - ``torch.Tensor`` – ``[H, W, 3]`` uint8, moved to CUDA if needed.

    Output modes:
        - ``container_format=None`` – raw Annex B. Per-frame packets are
          returned from :meth:`encode_frame`; the tail packet is returned from
          :meth:`end`.
        - ``container_format="mp4"`` / ``"matroska"`` / ... – the full Annex B
          stream is buffered and remuxed through PyAV on :meth:`end`.
    """

    def __init__(self, config: NvcCoderConfig):
        if not isinstance(config, NvcCoderConfig):
            raise TypeError(
                f"NvcCoder requires NvcCoderConfig, got {type(config).__name__}"
            )
        if config.frame_format not in ("rgb24", "bgr24"):
            raise ValueError(
                f"NvcCoder only supports rgb24/bgr24 frame_format, "
                f"got {config.frame_format!r}"
            )
        super().__init__(config)
        self._cuda_device_id = None
        self._torch_device = None
        self._cuda_context = None
        self._cuda_stream = None
        self._encoder = None

    def _set_log_level(self, level):
        # PyNvVideoCodec does not expose a shared logging control.
        return None

    def _ensure_cuda_binding(self) -> None:
        if self._torch_device is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("NvcCoder requires CUDA, but torch.cuda is unavailable")
        device_id = int(torch.cuda.current_device())
        cuda = _get_cuda_driver()
        _check_cuda_result(cuda.cuInit(0), "cuInit")
        torch_device = torch.device("cuda", device_id)
        with torch.cuda.device(device_id):
            torch.empty(0, device=torch_device)
            ctx = ctypes.c_void_p()
            _check_cuda_result(
                cuda.cuCtxGetCurrent(ctypes.byref(ctx)),
                "cuCtxGetCurrent",
            )
            if not ctx.value:
                primary_ctx = _get_primary_cuda_context(device_id)
                _check_cuda_result(
                    cuda.cuCtxSetCurrent(ctypes.c_void_p(primary_ctx)),
                    "cuCtxSetCurrent",
                )
                _check_cuda_result(
                    cuda.cuCtxGetCurrent(ctypes.byref(ctx)),
                    "cuCtxGetCurrent",
                )
            if not ctx.value:
                raise RuntimeError(
                    f"No active CUDA context found for device {device_id}"
                )
            stream = int(torch.cuda.current_stream(device_id).cuda_stream)
        self._cuda_device_id = device_id
        self._torch_device = torch_device
        self._cuda_context = int(ctx.value)
        self._cuda_stream = stream
        self.get_logger().info(
            "Binding NVENC to %s (context=%s, stream=%s)",
            _format_cuda_visible_mapping(device_id),
            hex(self._cuda_context),
            hex(self._cuda_stream),
        )

    def set_output(self, file_path: PathLike = ""):
        self._file_path = str(file_path) if file_path else ""
        self._encoder = None
        self._width = None
        self._height = None
        self._nvenc_bytes = bytearray()
        if self.config.container_format is None:
            self._outbuf = None
            self._container = None
        else:
            self._outbuf = None if self._file_path else BytesIO()
            self._container = None

    def configure_stream(
        self,
        width: int,
        height: int,
        pix_fmt: Literal["yuv420p", "rgb24"] = "yuv420p",
        **codec_kwargs,
    ):
        if pix_fmt not in ("yuv420p", "rgb24"):
            raise ValueError(
                f"NvcCoder only supports yuv420p/rgb24 pix_fmt, got {pix_fmt!r}"
            )
        cfg = self.config
        fps = codec_kwargs.pop("fps", cfg.fps)
        bitrate = codec_kwargs.pop("bitrate", cfg.bitrate)
        gop = codec_kwargs.pop("gop", cfg.gop or fps)
        preset = codec_kwargs.pop("preset", cfg.preset)
        rc = codec_kwargs.pop("rc", cfg.rc)
        tuninginfo = codec_kwargs.pop("tuninginfo", cfg.tuninginfo)
        bf = codec_kwargs.pop("bf", cfg.bf)
        if codec_kwargs:
            self.get_logger().warning(
                "Ignoring unsupported NVENC kwargs: %s", sorted(codec_kwargs)
            )

        if self._encoder is not None:
            try:
                tail = self._encoder.EndEncode()
                if tail:
                    self._nvenc_bytes.extend(bytes(tail))
            except Exception:
                self.get_logger().exception(
                    "NVENC EndEncode failed during reconfigure"
                )
            self._encoder = None
            gc.collect()

        self._ensure_cuda_binding()
        self._encoder = nvc.CreateEncoder(
            width,
            height,
            "NV12",
            False,
            gpuid=self._cuda_device_id,
            cudacontext=self._cuda_context,
            cudastream=self._cuda_stream,
            codec=cfg.nvenc_codec,
            preset=preset,
            tuninginfo=tuninginfo,
            rc=rc,
            bitrate=bitrate,
            fps=fps,
            gop=gop,
            bf=bf,
        )
        self._width = width
        self._height = height
        self._fps = fps
        self._configured = True

    def _coerce_frame(self, frame: FrameInput) -> torch.Tensor:
        """Normalize input to contiguous ``[H, W, 3]`` uint8 CUDA tensor in RGB order."""
        self._ensure_cuda_binding()
        if isinstance(frame, torch.Tensor):
            arr = (
                frame
                if frame.device == self._torch_device
                else frame.to(self._torch_device, non_blocking=False)
            )
        else:
            if self._preprocess is None:
                self._set_frame_type(frame)
            decoded = self._preprocess(frame)
            if not isinstance(decoded, np.ndarray):
                raise TypeError(
                    f"Unexpected preprocess output type: {type(decoded).__name__}"
                )
            arr = torch.from_numpy(decoded).to(self._torch_device, non_blocking=False)

        if arr.dtype != torch.uint8:
            raise TypeError(f"Frame dtype must be uint8, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"Frame must be [H, W, 3] uint8, got shape {tuple(arr.shape)}"
            )

        if self._frame_format == "bgr24":
            arr = arr.flip(-1)
        return arr.contiguous()

    def encode_frame_blocking(
        self,
        frame: FrameInput,
        timestamp: int,
        ns_to_base: bool = False,
    ) -> List[Packet]:
        assert isinstance(timestamp, int), "Timestamp must be an integer"
        timestamp = timestamp // self._ns2base if ns_to_base else timestamp
        if self._start_time is None:
            if timestamp < 0:
                raise ValueError("Timestamp must not be negative")
            self._start_time = timestamp

        rgb = self._coerce_frame(frame)

        if not self._configured:
            h, w = rgb.shape[:2]
            self.configure_stream(w, h)
        elif rgb.shape[:2] != (self._height, self._width):
            raise ValueError(
                f"Frame size {tuple(rgb.shape[:2])} != configured "
                f"({self._height}, {self._width})"
            )

        last_time = self._last_time
        if timestamp <= last_time:
            mode = self.config.non_monotonic_mode
            error_msg = (
                f"Frame timestamp {timestamp} is not greater than "
                f"last timestamp {last_time}"
            )
            if mode is NonMonotonicTimeMode.RAISE:
                raise ValueError(error_msg)
            if self.config.non_monotonic_log:
                self.get_logger().warning("%s, %s", error_msg, mode)
            if mode is NonMonotonicTimeMode.DROP:
                return []
            if mode is NonMonotonicTimeMode.ADJUST:
                timestamp = last_time + max(self._time_base.denominator // 1000, 1)
        self._last_time = timestamp

        nv12 = rgb_to_nv12(rgb)
        torch.cuda.synchronize(device=self._torch_device)
        ensure_torch_dlpack_positional_stream_compat()
        pkt = self._encoder.Encode(nv12)
        if not pkt:
            return []
        packet_bytes = bytes(pkt)
        self._nvenc_bytes.extend(packet_bytes)
        return [packet_bytes]

    def _end(self) -> List[Packet]:
        tail_bytes = b""
        if self._encoder is not None:
            try:
                tail = self._encoder.EndEncode()
                if tail:
                    tail_bytes = bytes(tail)
            except Exception:
                self.get_logger().exception("NVENC EndEncode failed in _end()")
            self._encoder = None
            gc.collect()

        if tail_bytes:
            self._nvenc_bytes.extend(tail_bytes)

        if self.config.container_format is None:
            return [tail_bytes] if tail_bytes else []

        self._remux_to_container()
        return []

    def _remux_to_container(self) -> None:
        """Remux the buffered Annex B stream into the configured container via PyAV.

        Raw h264/hevc demuxers emit packets without pts/dts, so timestamps are
        synthesized from ``fps`` (the same hint NVENC was configured with).
        """
        if not self._nvenc_bytes:
            return
        target = self._file_path or self._outbuf
        if target is None:
            raise RuntimeError("No output target for container mode")

        import fractions

        src_format = self.config.nvenc_codec
        time_base = fractions.Fraction(1, self._fps * 1000)
        duration = int(1 / float(time_base) / self._fps)
        with av.open(
            BytesIO(bytes(self._nvenc_bytes)), "r", format=src_format
        ) as src:
            src_stream = src.streams.video[0]
            with av.open(
                target, "w", format=self.config.container_format
            ) as container:
                out_stream = container.add_stream_from_template(src_stream)
                out_stream.time_base = time_base
                index = 0
                for packet in src.demux(src_stream):
                    if not packet.size:
                        continue
                    packet.stream = out_stream
                    packet.time_base = time_base
                    packet.pts = index * duration
                    packet.dts = packet.pts
                    packet.duration = duration
                    container.mux(packet)
                    index += 1

    def _close(self):
        if self._encoder is not None:
            try:
                self._encoder.EndEncode()
            except Exception:
                self.get_logger().exception(
                    "NVENC EndEncode failed during close()"
                )
            self._encoder = None
            gc.collect()
        if getattr(self, "_outbuf", None) is not None:
            self._outbuf.close()
            self._outbuf = None
        self._nvenc_bytes = bytearray()
        self._container = None
