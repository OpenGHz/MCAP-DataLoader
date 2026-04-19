"""耗时分析 test_h264.py 的各个阶段."""
from pathlib import Path
import time

import PyNvVideoCodec as nvc
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from tests.test_h264 import rgb_to_nv12, make_encoder


def sync():
    torch.cuda.synchronize()


def main():
    T, B, N, H, W = 60, 2, 3, 256, 256
    fps, bitrate = 30, 2_000_000
    n_streams = B * N

    torch.manual_seed(0)
    # 数据生成放计时之外
    frame_list = [
        (torch.rand(B, N, H, W, 3, device="cuda") * 255).to(torch.uint8)
        for _ in range(T)
    ]
    sync()

    # 1. 创建编码器
    t0 = time.perf_counter()
    encoders = [make_encoder(W, H, fps, bitrate) for _ in range(n_streams)]
    sync()
    t_create = time.perf_counter() - t0

    # 2. 纯 RGB->NV12 耗时 (T * n_streams 帧)
    sync(); t0 = time.perf_counter()
    nv12_frames = []
    for t in range(T):
        batch = frame_list[t].reshape(n_streams, H, W, 3)
        for s in range(n_streams):
            nv12_frames.append(rgb_to_nv12(batch[s]))
    sync()
    t_convert = time.perf_counter() - t0

    # 3. 纯 Encode 耗时 (用上面已转好的 NV12)
    sync(); t0 = time.perf_counter()
    bitstreams = [bytearray() for _ in range(n_streams)]
    idx = 0
    for t in range(T):
        for s in range(n_streams):
            pkt = encoders[s].Encode(nv12_frames[idx])
            if pkt:
                bitstreams[s].extend(bytes(pkt))
            idx += 1
    t_encode = time.perf_counter() - t0

    # 4. EndEncode 刷尾
    t0 = time.perf_counter()
    for s in range(n_streams):
        tail = encoders[s].EndEncode()
        if tail:
            bitstreams[s].extend(bytes(tail))
    t_flush = time.perf_counter() - t0

    total_frames = T * n_streams
    print(f"参数: T={T}, B={B}, N={N}, H={H}, W={W}, 总帧数={total_frames}")
    print(f"  1) 创建 {n_streams} 个 encoder : {t_create*1000:8.1f} ms "
          f"({t_create/n_streams*1000:.1f} ms/enc)")
    print(f"  2) RGB->NV12  ({total_frames} 帧): {t_convert*1000:8.1f} ms "
          f"({t_convert/total_frames*1000:.2f} ms/帧)")
    print(f"  3) NVENC Encode ({total_frames} 帧): {t_encode*1000:8.1f} ms "
          f"({t_encode/total_frames*1000:.2f} ms/帧)")
    print(f"  4) EndEncode 刷尾           : {t_flush*1000:8.1f} ms")
    tot = t_create + t_convert + t_encode + t_flush
    print(f"  总计                        : {tot*1000:8.1f} ms")
    print()
    # 单路吞吐
    fps_per_stream = T / (t_encode / n_streams)  # 平均给每路的时间
    print(f"聚合 NVENC 吞吐: {total_frames/t_encode:.1f} fps")
    print(f"单路吞吐 (摊分): {fps_per_stream:.1f} fps")


if __name__ == "__main__":
    main()
