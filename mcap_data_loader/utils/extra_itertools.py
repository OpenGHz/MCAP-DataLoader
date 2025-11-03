from itertools import chain, islice, tee
from more_itertools import consume, first
from collections import deque
from collections.abc import Iterator, Generator
from typing import Any, Iterable, Callable, TypeVar, Generic, Tuple, List


T = TypeVar("T")


def epairwise(
    iterable: Iterable[T],
    gap: int = 0,
    fillvalue: Any = ...,
    fill_with_last: bool = False,
) -> Generator[Tuple[T, T]]:
    a, b = tee(iterable)
    consume(b, gap + 1)
    if not fill_with_last:
        if fillvalue is ...:
            return zip(a, b)
        return zip(a, chain(b, (fillvalue,) * (gap + 1)))

    def fill_last_gen(it: Iterable[T]) -> Generator[T]:
        for item in it:
            yield item
        while True:
            yield item

    return zip(a, fill_last_gen(b))


def ewindowed(
    seq: Iterable[T],
    n: int,
    fillvalue: Any = None,
    step: int = 1,
    fill_with_last: bool = False,
) -> Generator[Tuple[T, ...]]:
    """Enhanced version of `more_itertools.windowed`: When `fill_with_last`
    is True, starting from the first element equal to `fillvalue`, it and
    all elements to its right are replaced with the left element of that element
    """
    # TODO: optimize (ref to epairwise)
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
    iterable: Iterable[T],
    past_num: int,
    future_num: int,
    fillvalue: Any = None,
    step: int = 1,
    fill_with_last: bool = False,
) -> Generator[Tuple[Tuple[T, ...], Tuple[T, ...]]]:
    """Generate pairs of (past, future) windows from the iterable.
    Each past window contains `past_num + 1` elements (including the current element),
    and each future window contains `future_num` elements. The total iteration steps
    equal to the length of the iterable when `step` is 1.
    """
    if isinstance(iterable, Iterator):
        raise ValueError("iterable must be a reusable iterable, not an iterator")
    try:
        first = next(iter(iterable))
    except StopIteration:
        return ()

    padded = chain([first] * past_num, iterable, [None] * future_num)

    windows = ewindowed(
        padded, past_num + future_num + 1, fillvalue, step, fill_with_last
    )

    for win in windows:
        past = win[: past_num + 1]
        future = win[past_num:]
        yield past, future


class Reusablizer(Generic[T]):
    # since the func return type is usually unknown when
    # passed to __init__, we don't annotate the return with T
    def __init__(self, func: Callable[..., Iterable]):
        self.gen_func = func

    def __call__(self, *args, **kwds) -> "Reusablizer[T]":
        self.args = args
        self.kwargs = kwds
        return self

    def __iter__(self) -> Iterator[T]:
        return self.gen_func(*self.args, **self.kwargs)


def take_skip(
    lst: Iterable[T], N: int, M: int, sort: bool = False
) -> Tuple[List[T], List[T]]:
    """Take N elements, skip M elements, repeat until the list is exhausted.
    Return two lists: taken elements and skipped elements.
    If sort is True, sort the two lists before returning.
    """
    if N <= 0 or M < 0:
        raise ValueError("N must be positive and M must be non-negative.")
    taken = []
    skipped = []
    for i in range(N):
        taken.extend(lst[i :: N + M])
    for j in range(M):
        skipped.extend(lst[N + j :: N + M])
    if sort:
        taken.sort()
        skipped.sort()
    return taken, skipped


def first_recursive(iterable: Iterable, depth: int = 1) -> Any:
    """Get the first element from a nested iterable structure up to a specified depth."""
    current = iterable
    if depth < 0:
        while isinstance(current, Iterable):
            current = first(current)
    else:
        for _ in range(depth):
            current = first(current)
    return current


if __name__ == "__main__":
    # import time

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

    # max_steps = 20
    # iterable = range(10)
    # # iterable = iter(range(10))  # raise an error
    # start = time.perf_counter()
    # cnt = 0
    # for past, future in past_future(iterable, 2, 3, None, 1, True):
    #     print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
    #     print(f"{past=}, {future=}")
    #     cnt += 1
    #     print(f"step: {cnt}/{max_steps}")
    #     start = time.perf_counter()

    # from itertools import pairwise

    # iterable = (f"{i}" for i in range(10))
    # it_reusable = Reusablizer[str](pairwise)(iterable)
    # for item in it_reusable:
    #     print(item)

    # iterable = range(10)
    # for a, b in epairwise(iterable, 2, fill_with_last=True):
    #     print(f"{a=}, {b=}")
    # for a, b in epairwise(iterable, 2, fill_with_last=False):
    #     print(f"{a=}, {b=}")
    # for a, b in epairwise(iterable, 2, None):
    #     print(f"{a=}, {b=}")

    # lis = range(17)
    # print(take_skip(lis, 2, 3, True))  # Example usage

    nested = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    print(
        first_recursive(nested, depth=0)
    )  # Output: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    print(first_recursive(nested, depth=1))  # Output: [[1, 2], [3, 4]]
    print(first_recursive(nested, depth=2))  # Output: [1, 2]
    print(first_recursive(nested, depth=-1))  # Output: 1
