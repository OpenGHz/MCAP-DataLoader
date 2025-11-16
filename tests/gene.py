from typing import Generator


def accumulator(initial: int) -> Generator[int, int, str]:
    total = initial
    while True:
        value: int = yield total  # YieldType=int, SendType=int
        if value is None:
            break
        total += value
    return f"Sum={total}"  # ReturnType=str


g = accumulator(1)
# print(next(g))  # → 0 (int)
g.send(None)  # Prime the generator
print(g.send(10))  # → 10 (int)
print(g.send(20))  # → 30 (int)

try:
    g.send(None)
except StopIteration as e:
    msg: str = e.value  # "Sum=30"
