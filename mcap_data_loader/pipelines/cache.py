from pydantic import BaseModel, NonNegativeInt
from collections.abc import Generator
from collections import deque
from mcap_data_loader.pipelines.basis import Pipe, T
from typing import Optional


class CacheConfig(BaseModel, frozen=True):
    """Configuration for Cache pipeline."""

    maxlen: Optional[NonNegativeInt] = None
    """Maximum length of the cache. If None, the cache can grow indefinitely."""


class Cache(Pipe[T]):
    """Cache items from an iterable up to a specified maximum length."""

    def __init__(self, config: CacheConfig) -> None:
        self._cache = deque(maxlen=config.maxlen)
        self._length = None

    def __iter__(self) -> Generator[T]:
        if self._length is None:
            index = -1
            for index, item in enumerate(self._iterable):
                self._cache.append(item)
                yield item
            self._length = index + 1
        else:
            iterator = iter(self._iterable)
            for _ in range(self._length - len(self._cache)):
                yield next(iterator)
            for item in self._cache:
                yield item

    def __len__(self) -> int:
        return self._length if self._length is not None else len(list(self.__iter__()))


if __name__ == "__main__":
    import time

    base = range(10)

    class SlowIterable:
        def __iter__(self):
            for item in base:
                time.sleep(0.1)  # Simulate a slow data source
                yield item

    data = SlowIterable()
    data_list = list(base)  # Exhaust the generator for testing

    for maxlen in [None, 3, 5]:
        print(f"\nTesting Cache with maxlen={maxlen}")
        # NOTE: The list method first calls len to get the length, and then iterates. The first call to len requires one iteration, and after getting len, list iterates again, thus increasing the total time.
        cached = Cache(CacheConfig(maxlen=maxlen))(data)
        start = time.perf_counter()
        assert list(cached) == data_list
        print(f"First iteration took {time.perf_counter() - start:.2f} seconds")
        start = time.perf_counter()
        assert list(cached) == data_list
        print(f"Second iteration took {time.perf_counter() - start:.2f} seconds")

    print("\nAll tests passed.")
