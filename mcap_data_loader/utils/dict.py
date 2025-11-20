from typing import Tuple, TypeVar
from collections.abc import Mapping, Iterable


T = TypeVar("T")


def iterable2dict(iterable: Iterable[T]) -> Mapping[int, T]:
    """Convert an iterable to a dictionary with integer keys."""
    if isinstance(iterable, Mapping):
        return iterable
    return {i: item for i, item in enumerate(iterable)}


def dict2tuple(d: Mapping[int, T]) -> Tuple[T, ...]:
    """Convert a dictionary with integer keys to an iterable."""
    return tuple(d[i] for i in range(len(d)))
