#!/usr/bin/env python3
# bench_astype.py
import timeit
import torch
import array_api_compat

# ---------------------- 配置区 ----------------------
N = 1_000_000  # tensor 元素个数
loops = 200  # timeit 重复次数
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------


def make_tensor(dtype, dev):
    """构造一个 1-D tensor 并保证在指定 device 上"""
    t = torch.randn(N, dtype=dtype)
    return t.to(dev)


def bench_case(name, tensor, target_dtype, target_device):
    """测一次 xp.astype 耗时"""
    xp = array_api_compat.get_namespace(tensor)  # 自动识别 PyTorch

    def stmt():
        return xp.astype(tensor, target_dtype, device=target_device, copy=False)

    t = timeit.timeit(stmt, number=loops) / loops
    print(f"{name:40s}  {t * 1e6:8.2f} µs")


def main():
    print(f"Device used: {device}")
    print("-" * 60)
    src = make_tensor(torch.float32, device)

    # 1. 类型一致 + 设备一致
    bench_case("same dtype + same device", src, torch.float32, device)

    # 2. 类型不一致 + 设备一致
    bench_case("diff dtype + same device", src, torch.int32, device)

    # 3. 类型一致 + 设备不一致 (CPU↔CUDA)
    other_dev = "cpu" if device == "cuda" else "cuda"
    if torch.cuda.is_available() or other_dev == "cpu":
        bench_case("same dtype + diff device", src, torch.float32, other_dev)

        # 4. 类型不一致 + 设备不一致
        bench_case("diff dtype + diff device", src, torch.int16, other_dev)

    def pass_through():
        return src

    t = timeit.timeit(pass_through, number=loops) / loops
    print(f"pass through  {t * 1e6:8.2f} µs")


if __name__ == "__main__":
    main()
