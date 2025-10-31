import timeit

# 测试 bool 判断
time_bool = timeit.timeit('if flag: pass', setup='flag = True', number=10_000_000)

# 测试 isinstance
time_isinstance = timeit.timeit('isinstance(x, dict)', setup='x = {}', number=10_000_000)

print(f"Bool check:     {time_bool:.3f} s")
print(f"isinstance:     {time_isinstance:.3f} s")
print(f"isinstance is ~{time_isinstance / time_bool:.1f}x slower")