import time


def test_case1(dict1, dict2):
    dict_new = {}
    for k, v in dict1.items():
        dict_new[k] = not v
    dict_new.update(dict2)
    return dict_new


def test_case2(dict1, dict_all):
    # 注意：不能直接改原 dict_all，需先复制（否则污染后续测试）
    for key in dict1:
        dict_all[key] = not dict_all[key]
    return dict_all


# 构造测试数据
def make_data(n1=1000, n2=1000):
    dict1 = {f"k{i}": True for i in range(n1)}  # 字符串值
    dict2 = {f"x{i}": i for i in range(n2)}  # 数值（不参与 *2）
    dict_all = dict1 | dict2
    return dict1, dict2, dict_all


def timeit(func, *args, repeat=1000):
    start = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    end = time.perf_counter()
    return (end - start) / repeat * 1e6  # µs


if __name__ == "__main__":
    # 场景1：小字典
    d1, d2, da = make_data(10, 10)
    t1 = timeit(test_case1, d1, d2)
    t2 = timeit(test_case2, d1, da)
    print(
        f"小字典 (10+10): Case1={t1:.2f}µs, Case2={t2:.2f}µs → Case2 快 {t1 / t2:.2f}x"
    )

    # 场景2：dict2 很大
    d1, d2, da = make_data(10, 10000)
    t1 = timeit(test_case1, d1, d2)
    t2 = timeit(test_case2, d1, da)
    print(
        f"大 dict2 (10+10k): Case1={t1:.2f}µs, Case2={t2:.2f}µs → Case2 快 {t1 / t2:.2f}x"
    )

    # 场景3：dict1 很大
    d1, d2, da = make_data(10000, 10)
    t1 = timeit(test_case1, d1, d2)
    t2 = timeit(test_case2, d1, da)
    print(
        f"大 dict1 (10k+10): Case1={t1:.2f}µs, Case2={t2:.2f}µs → Case2 快 {t1 / t2:.2f}x"
    )
