import time

# 配置
N = 1_000_000  # 字典大小
REPEAT = 10  # 重复轮数取平均

# 构造字典：key 为字符串，value 为整数
data = {f"key_{i}": i * 2 for i in range(N)}


def method_items():
    total = 0
    for k, v in data.items():
        total += v  # 直接使用 v
    return total


def method_key_lookup():
    total = 0
    for k in data:
        total += data[k]  # 再次哈希查找
    return total


def timeit(func, repeat=REPEAT):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time, result


if __name__ == "__main__":
    time_items, res1 = timeit(method_items)
    time_lookup, res2 = timeit(method_key_lookup)

    assert res1 == res2, "结果不一致！"

    print(f"字典大小: {N:,}")
    print(f"{'items() 方法':<20}: {time_items:.6f} 秒")
    print(f"{'key + d[key] 方法':<20}: {time_lookup:.6f} 秒")
    speedup = time_lookup / time_items if time_items > 0 else float("inf")
    print(f"加速比: {speedup:.2f}x")
