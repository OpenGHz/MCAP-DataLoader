"""
B*N 路独立 H264 视频编码示例 (GPU NVENC)

输入: List[Tensor[B, N, H, W, 3]] 长度 T, uint8, CUDA
输出: B*N 个独立 .h264 文件, 每个 T 帧

依赖:
    pip install PyNvVideoCodec torch
"""

import logging
from pathlib import Path
from typing import Optional

import PyNvVideoCodec as nvc
import torch

_logger = logging.getLogger(__name__)


def rgb_to_nv12(rgb: torch.Tensor) -> torch.Tensor:
    """[H, W, 3] uint8 RGB -> [H*3/2, W] uint8 NV12, 全程 GPU."""
    assert rgb.dtype == torch.uint8 and rgb.shape[-1] == 3
    H, W, _ = rgb.shape
    assert H % 2 == 0 and W % 2 == 0, "H/W 必须为偶数"

    rgb_f = rgb.float()
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

    y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp(0, 255)
    u = (-0.148 * r - 0.291 * g + 0.439 * b + 128.0).clamp(0, 255)
    v = (0.439 * r - 0.368 * g - 0.071 * b + 128.0).clamp(0, 255)

    u_sub = u.reshape(H // 2, 2, W // 2, 2).mean(dim=(1, 3))
    v_sub = v.reshape(H // 2, 2, W // 2, 2).mean(dim=(1, 3))
    uv = torch.stack([u_sub, v_sub], dim=-1).reshape(H // 2, W)

    return torch.cat([y, uv], dim=0).to(torch.uint8).contiguous()


class H264StreamEncoder:
    """单路 H264 GPU 编码器.

    生命周期:
        __init__()                -> 不创建底层 encoder, 零 GPU 代价
        configure(W, H, fps, br)  -> 首次创建; finalize 之后必定重建 (保证下一段以 IDR 起始, 可独立解码);
                                     若仅码率/帧率变化且分辨率不变, 走 Reconfigure 不打断流
        encode(rgb_hwc)           -> 编码一帧, 追加到当前 bitstream
        finalize(path)            -> 刷尾, 落盘, 清空 bitstream, 标记编码器待重建
        close()                   -> 释放底层 encoder
    """

    def __init__(self):
        self._encoder: Optional[object] = None
        self._W: Optional[int] = None
        self._H: Optional[int] = None
        self._fps: Optional[int] = None
        self._bitrate: Optional[int] = None
        self._codec: str = "h264"
        self._bitstream = bytearray()
        self._needs_recreate = True

    def configure(
        self, W: int, H: int, fps: int = 30, bitrate: int = 4_000_000
    ) -> None:
        """首次创建; finalize 之后或分辨率变化时重建, 否则走 Reconfigure."""
        need_recreate = (
            self._encoder is None
            or self._needs_recreate
            or W != self._W
            or H != self._H
        )

        if need_recreate:
            if self._encoder is not None:
                tail = self._encoder.EndEncode()
                if tail:
                    self._bitstream.extend(bytes(tail))
                self._encoder = None

            self._encoder = nvc.CreateEncoder(
                W,
                H,
                "NV12",
                False,
                codec=self._codec,
                preset="P4",
                tuninginfo="high_quality",
                rc="vbr",
                bitrate=bitrate,
                fps=fps,
                gop=fps,
            )
        else:
            params = self._encoder.GetEncodeReconfigureParams()
            params.averageBitrate = bitrate
            params.frameRateNum = fps
            params.frameRateDen = 1
            ok = self._encoder.Reconfigure(params)
            if not ok:
                raise RuntimeError("NVENC Reconfigure 失败")

        self._W, self._H, self._fps, self._bitrate = W, H, fps, bitrate
        self._bitstream = bytearray()
        self._needs_recreate = False

    def encode(self, rgb: torch.Tensor) -> None:
        """编码一帧 RGB [H, W, 3] uint8 CUDA 张量."""
        assert self._encoder is not None, "请先调用 configure()"
        assert rgb.shape[:2] == (self._H, self._W), (
            f"帧尺寸 {tuple(rgb.shape[:2])} 与 configure 的 ({self._H}, {self._W}) 不符"
        )
        nv12 = rgb_to_nv12(rgb)
        pkt = self._encoder.Encode(nv12)
        if pkt:
            self._bitstream.extend(bytes(pkt))

    def finalize(self, out_path: Optional[Path] = None) -> bytes:
        """刷出残留包, 返回当前 stream 的完整字节流 (并可选写文件).

        调用后会标记 encoder 待重建, 下次 configure 强制创建新实例,
        以保证下一段码流以 IDR 起始, 可独立解码.
        """
        assert self._encoder is not None, "请先调用 configure()"
        tail = self._encoder.EndEncode()
        if tail:
            self._bitstream.extend(bytes(tail))
        data = bytes(self._bitstream)
        if out_path is not None:
            Path(out_path).write_bytes(data)
        self._bitstream = bytearray()
        self._needs_recreate = True
        return data

    def close(self) -> None:
        if self._encoder is not None:
            try:
                self._encoder.EndEncode()
            except Exception:
                _logger.exception("NVENC EndEncode failed during close()")
            self._encoder = None
            self._needs_recreate = True


def encode_streams(
    frame_list: list[torch.Tensor],
    out_dir: str | Path,
    fps: int = 30,
    bitrate: int = 4_000_000,
) -> list[Path]:
    """示例: B*N 个编码器实例, 每路独立喂帧."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = len(frame_list)
    assert T > 0
    B, N, H, W, C = frame_list[0].shape
    assert C == 3
    for t, x in enumerate(frame_list):
        assert x.shape == (B, N, H, W, C), f"frame {t} shape mismatch"
        assert x.is_cuda, "输入必须在 GPU 上"

    encoders = [[H264StreamEncoder() for _ in range(N)] for _ in range(B)]
    for b in range(B):
        for n in range(N):
            encoders[b][n].configure(W, H, fps=fps, bitrate=bitrate)

    import time

    start = time.perf_counter()

    for t in range(T):
        batch = frame_list[t]
        for b in range(B):
            for n in range(N):
                encoders[b][n].encode(batch[b, n])

    paths: list[Path] = []
    for b in range(B):
        for n in range(N):
            p = out_dir / f"stream_b{b}_n{n}.h264"
            data = encoders[b][n].finalize(p)
            encoders[b][n].close()
            paths.append(p)
            print(f"wrote {p}  ({len(data)} bytes)")
    end = time.perf_counter()
    print(f"总耗时: {end - start:.2f} 秒")
    return paths


if __name__ == "__main__":
    T, B, N, H, W = 60, 2, 3, 256, 256
    torch.manual_seed(0)
    frame_list = [
        (torch.rand(B, N, H, W, 3, device="cuda") * 255).to(torch.uint8)
        for _ in range(T)
    ]
    out_dir = Path(__file__).parent / "out_videos"
    encode_streams(frame_list, out_dir, fps=30, bitrate=2_000_000)
    print(f"\n共 {B * N} 个 H264 流, 每个 {T} 帧, 输出到 {out_dir}")
