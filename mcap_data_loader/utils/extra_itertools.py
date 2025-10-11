from itertools import chain, tee, islice
from collections import deque
from typing import Any, Iterable


def ewindowed(
    seq: Iterable,
    n: int,
    fillvalue: Any = None,
    step: int = 1,
    fill_with_last: bool = False,
):
    """Enhanced version of `more_itertools.windowed`: When `fill_with_last`
    is True, starting from the first element equal to `fillvalue`, it and
    all elements to its right are replaced with the left element of that element
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        yield ()
        return
    if step < 1:
        raise ValueError("step must be >= 1")

    iterable = iter(seq)

    # Generate first window
    window = deque(islice(iterable, n), maxlen=n)

    # Deal with the first window not being full
    if not window:
        return
    elif fill_with_last:
        if window[0] != fillvalue:
            for index in range(len(window)):
                item = window[-1]
                if item != fillvalue:
                    window.extend(item for _ in range(index))
                    break
                last_val = window.pop()

    if len(window) < n:
        # Use last value for padding if requested
        if fill_with_last:
            last_val = window[-1]
            yield tuple(window) + ((last_val,) * (n - len(window)))
        else:
            yield tuple(window) + ((fillvalue,) * (n - len(window)))
        return
    yield tuple(window)

    def iter_wrapper():
        last_val = None
        for item in iterable:
            if fill_with_last and item == fillvalue:
                yield last_val
            else:
                last_val = item
                yield item
        if fill_with_last:
            fillval = last_val
        else:
            fillval = fillvalue
        padding = (fillval for _ in range(n - 1 if step >= n else step - 1))
        for pad_val in padding:
            yield pad_val

    filler = map(window.append, iter_wrapper())

    for _ in islice(filler, step - 1, None, step):
        yield tuple(window)


def past_future(
    iterable: Iterable,
    past_num: int,
    future_num: int,
    fillvalue: Any = None,
    step: int = 1,
    fill_with_last: bool = False,
):
    """Generate pairs of (past, future) windows from the iterable.
    Each past window contains `past_num + 1` elements (including the current element),
    and each future window contains `future_num` elements. The total iteration steps
    equal to the length of the iterable when `step` is 1.
    """
    it1, it2 = tee(iterable)
    try:
        first = next(it1)
    except StopIteration:
        return ()

    padded = chain([first] * past_num, it2, [None] * future_num)

    windows = ewindowed(
        padded, past_num + future_num + 1, fillvalue, step, fill_with_last
    )

    for win in windows:
        past = win[: past_num + 1]
        future = win[past_num:]
        yield past, future


if __name__ == "__main__":
    import time

    # iterables = [range(2), [1, None], [None], chain(range(4), [None] * 10)]

    # for iterable in iterables:
    #     start = time.perf_counter()
    #     rounds = 3
    #     for window in windowed(iterable, 3, None, 1, True):
    #         print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
    #         print(window)
    #         start = time.perf_counter()
    #         rounds -= 1
    #         if rounds == 0:
    #             break
    #     print("===" * 10)

    max_steps = 20
    iterable = range(10)
    start = time.perf_counter()
    cnt = 0
    for past, future in past_future(iterable, 2, 3, None, 1, True):
        print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
        print(f"{past=}, {future=}")
        cnt += 1
        print(f"step: {cnt}/{max_steps}")
        start = time.perf_counter()
