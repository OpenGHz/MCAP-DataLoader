"""Discard the first N and last M elements from the data."""

from collections import deque
from collections.abc import Generator
from more_itertools import consume
from pydantic import BaseModel, NonNegativeInt
from mcap_data_loader.pipelines.basis import Pipe, T


class DropConfig(BaseModel, frozen=True):
    """Configuration for the Drop pipeline."""

    head: NonNegativeInt = 0
    """Number of items to discard from the start of the iterable."""

    tail: NonNegativeInt = 0
    """Number of items to discard from the end of the iterable."""


class Drop(Pipe[T]):
    """Drop the first *head* items and the last *tail* items from the iterable."""

    def __init__(self, config: DropConfig) -> None:
        self._head = config.head
        self._tail = config.tail

    def __iter__(self) -> Generator[T]:
        iterator = iter(self._iterable)
        # Skip items from the head eagerly using more-itertools.consume.
        consume(iterator, self._head)
        tail = self._tail
        if tail == 0:
            yield from iterator
        else:
            # Remaining items in the buffer correspond to the tail that should be dropped.
            buffer: deque[T] = deque()
            for item in iterator:
                buffer.append(item)
                if len(buffer) > tail:
                    yield buffer.popleft()


__all__ = ["Drop", "DropConfig"]


if __name__ == "__main__":
    iterable = range(10)

    config = DropConfig(head=3, tail=4)
    dropped = Drop(config)(iterable)

    assert tuple(dropped) == (3, 4, 5)
