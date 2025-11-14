import time
import numpy as np
from collections import ChainMap
import random

# 配置
N_KEYS = 10
ARRAY_SHAPE = (1080, 1920, 3)
REPEAT_MERGE = 100000  # 合并重复次数（用于平均）
REPEAT_READ = 1000000  # 每次读取重复次数
READ_TRIALS = 10  # 读取测试轮数（取平均）

# 创建大数组（避免每次创建开销）
dummy_array = np.random.randint(0, 256, size=ARRAY_SHAPE, dtype=np.uint8)

# 构造两个字典
keys1 = [f"key_{i}" for i in range(N_KEYS)]
keys2 = [
    f"key_{i + N_KEYS}" for i in range(N_KEYS)
]  # 无冲突，便于公平比较；也可设部分冲突
# 若想测试冲突覆盖，可让 keys2 = keys1[:5] + [f'key_new_{i}' for i in range(5)]

d1 = {k: dummy_array.copy() for k in keys1}
d2 = {k: dummy_array.copy() for k in keys2}

all_keys = list(d1.keys()) + list(d2.keys())


def timeit(func, repeat=1):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times), result


# ====== 方法 1: unpacking {**d1, **d2} ======
def merge_unpack():
    return {**d1, **d2}


# ====== 方法 2: dict(d1, **d2) —— 注意：仅当 d2 的 key 全为 str 时有效 ======
def merge_dict_kwargs():
    return dict(d1, **d2)


# ====== 方法 3: | 运算符（Python 3.9+） ======
def merge_union():
    return d1 | d2


def merge_union_on():
    global d1
    d1 |= d2
    return d1


# ====== 方法 4: copy + update ======
def merge_copy_update():
    new = d1.copy()
    new.update(d2)
    return new


# ====== 方法 5: 字典推导（d2 优先） ======
def merge_dict_comp():
    # 注意：这里实现的是“d2 覆盖 d1”，但只遍历 d1；需补全 d2 中独有的键
    merged = {k: d2.get(k, v) for k, v in d1.items()}
    for k, v in d2.items():
        if k not in merged:
            merged[k] = v
    return merged


# ====== 方法 6: ChainMap（不复制数据） ======
def merge_chainmap():
    return ChainMap(d2, d1)  # d2 优先


# ====== 读取测试函数 ======
def test_read_time(merged_obj, is_chainmap=False):
    total_time = 0.0
    for _ in range(READ_TRIALS):
        random.shuffle(all_keys)
        start = time.perf_counter()
        for k in all_keys * (REPEAT_READ // len(all_keys)):
            if is_chainmap:
                _ = merged_obj[k]
            else:
                _ = merged_obj[k]
        end = time.perf_counter()
        total_time += end - start
    return total_time / READ_TRIALS


# ====== 执行测试 ======
if __name__ == "__main__":
    methods = [
        ("{**d1, **d2}", merge_unpack, False),
        ("dict(d1, **d2)", merge_dict_kwargs, False),
        ("d1 | d2", merge_union, False),
        ("d1 |= d2", merge_union_on, False),
        ("copy + update", merge_copy_update, False),
        ("dict comprehension", merge_dict_comp, False),
        ("ChainMap", merge_chainmap, True),
    ]

    print(f"测试配置：每个字典 {N_KEYS} 个键，值为 shape={ARRAY_SHAPE} 的 np.array")
    print(
        f"合并重复次数: {REPEAT_MERGE}, 读取重复总次数: {REPEAT_READ} 次/轮 × {READ_TRIALS} 轮\n"
    )

    results = []

    for name, merge_func, is_chain in methods:
        # 合并耗时
        avg_merge_time, merged = timeit(merge_func, repeat=REPEAT_MERGE)

        # 读取耗时
        avg_read_time = test_read_time(merged, is_chainmap=is_chain)

        results.append((name, avg_merge_time, avg_read_time))
        print(
            f"{name:>20}: 合并耗时 = {avg_merge_time * 1e6:.2f} µs, 读取耗时 = {avg_read_time * 1e6:.2f} µs"
        )

    # 补充说明
    print("\n💡 注意：")
    print("- ChainMap 不复制数据，合并极快，但读取稍慢（因需链式查找）")
    print("- dict(d1, **d2) 要求 d2 的 key 必须是合法标识符（str 且非关键字等）")
    print("- 实际性能受 Python 版本、字典大小、键冲突情况影响")
