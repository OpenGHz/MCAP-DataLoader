from itertools import chain, islice, tee
from more_itertools import take
from collections import deque
from collections.abc import Iterator, Generator, Callable
from typing import Any, Iterable, Optional, TypeVar, Generic, Tuple, List, Union
from copy import copy
from functools import partial


T = TypeVar("T")
_marker = object()


def first(iterable: Iterable, default: Any = _marker):
    """Return the first item of an iterable or raise StopIteration if empty."""
    item = next(iter(iterable), default)
    if item is _marker:
        raise ValueError(
            "first() was called on an empty iterable, "
            "and no default value was provided."
        )
    return item


def consume_and_return(
    iterator: Iterator[T], n: Optional[int] = None, strict: bool = False
) -> T:
    """"""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        que = deque(iterator, maxlen=1)
        return que.pop()
    else:
        taken = take(n, iterator)
        if strict and len(taken) < n:
            raise ValueError("not enough values to consume")
        return taken[-1]


def epairwise(
    iterable: Iterable[T],
    gap: int = 0,
    fillvalue: Any = ...,
    fill_with_last: bool = False,
    strict: bool = False,
) -> Generator[Tuple[T, T]]:
    a, b = tee(iterable)
    last = consume_and_return(b, gap + 1, strict)
    if not fill_with_last:
        if fillvalue is ...:
            return zip(a, b)
        return zip(a, chain(b, (fillvalue,) * (gap + 1)))

    def fill_last_gen(it: Iterable[T], item) -> Generator[T]:
        for item in it:
            yield item
        while True:
            yield item

    return zip(a, fill_last_gen(b, last))


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
                window.pop()

    if len(window) < n:
        # Use last value for padding if requested
        if fill_with_last:
            yield tuple(window) + ((window[-1],) * (n - len(window)))
        else:
            yield tuple(window) + ((fillvalue,) * (n - len(window)))
        return
    yield tuple(window)

    step_minus_one = step - 1
    pad_count = n - 1 if step >= n else step_minus_one
    append_window = window.append
    use_last = fill_with_last

    def iter_wrapper():
        last_val = window[-1]
        for item in iterable:
            if use_last and item == fillvalue:
                yield last_val
            else:
                last_val = item
                yield item
        fillval = last_val if use_last else fillvalue
        for _ in range(pad_count):
            yield fillval

    counter = step_minus_one
    for value in iter_wrapper():
        append_window(value)
        if counter == 0:
            yield tuple(window)
            counter = step_minus_one
        else:
            counter -= 1


def past_future(
    iterable: Iterable[T],
    past_num: int,
    future_num: int,
    fillvalue: Any = None,
    step: int = 1,
    fill_with_last: bool = False,
    gap: int = 0,
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

    padded = chain([first] * past_num, iterable, [None] * (future_num + gap))

    windows = ewindowed(
        padded, past_num + gap + future_num + 1, fillvalue, step, fill_with_last
    )
    for win in windows:
        yield win[: past_num + 1], win[past_num + gap :]


class Reusablizer(Generic[T]):
    """A wrapper to make generator functions reusable as iterables."""

    # since the func return type is usually unknown when
    # passed to __init__, we don't annotate the return with T
    def __init__(self, func: Callable[..., Iterable[T]]):
        self._func = func
        self._partial = None

    def __call__(self, *args, **kwds) -> "Reusablizer[T]":
        if self._partial is not None:
            self = copy(self)
        self._partial = partial(self._func, *args, **kwds)
        return self

    def __iter__(self) -> Iterator[T]:
        result = self._partial()
        if not isinstance(result, Iterator):
            raise TypeError(
                f"The wrapped function {self._partial}"
                f" did not return an iterator: {result}"
            )
        return result

    def __repr__(self):
        cls_path = f"{self.__module__}.{self.__class__.__name__}"
        if self._partial is None:
            return f"{cls_path}: {self._func}"
        else:
            return f"{cls_path}: {self._partial}"


def take_skip(
    lst: Iterable[T], N: int, M: int, in_order: bool = False
) -> Tuple[List[T], List[T]]:
    """Take N elements, skip M elements, repeat until the list is exhausted.
    Args:
        lst: The input iterable.
        N: Number of elements to take.
        M: Number of elements to skip.
        in_order: Whether to take and skip elements in order or round-robin.
            If in_order is True, the elements are taken and skipped in the original order (slower).
            If in_order is False, the elements are taken and skipped in a round-robin fashion.
    Returns:
        Two lists: taken elements and skipped elements.
    """
    if N <= 0 or M < 0:
        raise ValueError("N must be positive and M must be non-negative.")
    taken = []
    skipped = []
    if in_order:
        index = 0
        length = len(lst)
        while index < length:
            taken.extend(lst[index : index + N])
            index += N
            skipped.extend(lst[index : index + M])
            index += M
    else:
        for i in range(N):
            taken.extend(lst[i :: N + M])
        for j in range(M):
            skipped.extend(lst[N + j :: N + M])
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


def first_recursive_true(iterable: Iterable, pred: Callable[[Any], bool]) -> Any:
    """Get the first element from a nested iterable structure that satisfies the predicate."""
    current = iterable
    while not pred(current):
        current = first(current)
    return current


def is_iterable_but_not_base(obj: Any, base_type: Union[tuple, type]) -> bool:
    """Judge whether an object is iterable but not of base types (like str, bytes)."""
    if isinstance(obj, base_type):
        return False
    return isinstance(obj, Iterable)


def recursive_map(
    func: Callable[..., Any],
    iterable: Iterable,
    depth: int = 0,
    base_type: Union[tuple, type] = (str, bytes),
    _recur_func: Optional[Callable[..., Any]] = None,
) -> Generator:
    """
    Lazily applies a function recursively to elements in a nested iterable structure,
    up to a specified depth. This will not flatten the structure; it preserves the original nesting.

    Args:
        func: The function to apply.
        iterable: The input iterable (e.g., list, tuple, generator).
        depth:
            - < 0: Apply `func` at all levels (unlimited recursion until leaf nodes).
            - 0: Apply `func` only to top-level items (default behavior).
            - 1: Apply `func` to second-level items (i.e., children of top-level items).
            - n: Apply `func` at the (n+1)-th nesting level.
        base_type: A tuple of types that should be treated as atomic (non-iterable),
                   even if they technically are iterable (e.g., str, bytes).
    Yields:
        Transformed items according to the specified depth, preserving structure lazily.
    """
    _recur_func = recursive_map if _recur_func is None else _recur_func
    if depth == 0:
        yield from map(func, iterable)
    else:
        for item in iterable:
            if depth > 0 or is_iterable_but_not_base(item, base_type):
                yield _recur_func(func, item, depth - 1, base_type, _recur_func)
            else:
                yield item


def recursive_map_reusable(
    func: Callable[..., Any],
    iterable: Iterable,
    depth: int = 0,
    base_type: Union[tuple, type] = (str, bytes),
) -> Union[Generator, Reusablizer]:
    """A reusable version of recursive_map for non-iterator iterables."""
    recur_map = (
        recursive_map if isinstance(iterable, Iterator) else Reusablizer(recursive_map)
    )
    return recur_map(func, iterable, depth, base_type, recur_map)


if __name__ == "__main__":
    # import time
    # import timeit

    # iterable = range(5)
    # assert first(iterable) == 0
    # assert first([], default=42) == 42
    # try:
    #     first([])
    # except ValueError as e:
    #     print(f"Caught expected exception: {e}")

    # assert consume_and_return(iter(iterable), 3) == 2
    # assert consume_and_return(iter(iterable)) == 4
    # assert consume_and_return(iter(iterable), 10) == 4
    # try:
    #     consume_and_return(iter(iterable), 10, True)  # should raise
    # except ValueError as e:
    #     print(f"Caught expected exception: {e}")

    # long = range(100000)
    # print(timeit.timeit(lambda: ewindowed(long, 25, None, 1, True), number=100 * 3000))

    # iterables = [range(2), [1, None], [None], chain(range(4), [None] * 10)]
    # for iterable in iterables:
    #     start = time.perf_counter()
    #     rounds = 3
    #     for window in ewindowed(iterable, 3, None, 1, True):
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

    from itertools import pairwise

    iterable = range(4)
    pairwise_reusable = Reusablizer[int](pairwise)
    print(pairwise_reusable)
    reusable = pairwise_reusable(iterable)
    print(reusable)
    assert not isinstance(reusable, Iterator)
    lis = list(reusable)
    assert lis == [(0, 1), (1, 2), (2, 3)]
    assert list(reusable) == lis  # reusable test
    new_iterable = pairwise_reusable(range(2))
    assert list(new_iterable) == [(0, 1)]
    assert list(reusable) == lis  # reusable test

    # iterable = range(10)
    # for a, b in epairwise(iterable, 2, fill_with_last=True):
    #     print(f"{a=}, {b=}")
    # for a, b in epairwise(iterable, 2, fill_with_last=False):
    #     print(f"{a=}, {b=}")
    # for a, b in epairwise(iterable, 2, None):
    #     print(f"{a=}, {b=}")

    # for in_order in [True, False]:
    #     print(f"in_order={in_order}")
    #     taken, skipped = take_skip(range(10), 3, 2, in_order=in_order)
    #     print("Taken:", taken)
    #     print("Skipped:", skipped)
    #     if in_order:
    #         assert taken == [0, 1, 2, 5, 6, 7]
    #         assert skipped == [3, 4, 8, 9]
    #     else:
    #         assert taken == [0, 5, 1, 6, 2, 7]
    #         assert skipped == [3, 8, 4, 9]

    # nested = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    # print(
    #     first_recursive(nested, depth=0)
    # )  # Output: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    # print(first_recursive(nested, depth=1))  # Output: [[1, 2], [3, 4]]
    # print(first_recursive(nested, depth=2))  # Output: [1, 2]
    # print(first_recursive(nested, depth=-1))  # Output: 1
    # print(first_recursive_true(nested, lambda x: isinstance(x, int)))  # Output: 1

    # def double(x) -> int:
    #     return x * 2

    # nested_numbers = [[1, 2], [3, 4, 5], [6]]

    # for depth in [-1, 0, 1]:
    #     print(f"Depth: {depth}")
    #     result = recursive_map(double, nested_numbers, depth=depth)
    #     for item in result:
    #         print(list(item))
    #     print("===" * 10)
