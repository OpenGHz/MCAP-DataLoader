import timeit
from dataclasses import dataclass


@dataclass(slots=True)
class ConfigDCS:
    param = 42


config = ConfigDCS()


def prepare():
    param = config.param
    for _ in range(1000):
        a = param


def no_prepare():
    for _ in range(1000):
        a = config.param


if __name__ == "__main__":
    n = 1000
    t1 = timeit.timeit("prepare()", globals=globals(), number=n)
    t2 = timeit.timeit("no_prepare()", globals=globals(), number=n)
    print(f"With prepare: {t1:.6f} seconds for {n} runs")
    print(f"Without prepare: {t2:.6f} seconds for {n} runs")
