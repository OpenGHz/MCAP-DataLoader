from pydantic import BaseModel, ConfigDict
from collections.abc import Mapping, Generator
from collections import ChainMap
from mcap_data_loader.pipelines.basis import Pipe, T
from typing import Literal


Methods = Literal["auto", "ChainMap", "+", "|", "none"]


class MergeConfig(BaseModel, frozen=True):
    """Configuration for Merge pipeline."""

    model_config = ConfigDict(extra="forbid")

    method: Methods = "auto"
    """Method to use for merging items."""
    replace: bool = False
    """Whether allow to replace existing items with new ones when merging."""


class Merge(Pipe[T]):
    """Merge multiple iterables' items into one by applying a specified method."""

    def __init__(self, config: MergeConfig) -> None:
        replace = config.replace
        self._methods = {
            "ChainMap": lambda items: ChainMap(*items),
            "+": self._sum_replace if replace else self._sum,
            "|": self._or_replace if replace else self._or,
            "none": lambda items: items,
        }
        self._method = config.method
        self._first_call = True

    def _sum_replace(self, items):
        base = items[0]
        for item in items[1:]:
            base += item
        return base

    def _sum(self, items):
        base = self._item_type()
        for item in items:
            base += item
        return base

    def _or_replace(self, items):
        base = items[0]
        for item in items[1:]:
            base |= item
        return base

    def _or(self, items):
        base = self._item_type()
        for item in items:
            base |= item
        return base

    def __iter__(self) -> Generator[T]:
        if self._first_call and self._method == "auto":
            first = next(zip(*self._iterable))
            self._item_type = type(first[0])
            if not all(isinstance(item, self._item_type) for item in first):
                raise ValueError(
                    f"All items in the first iterable must be of type {self._item_type}, "
                    f"but got {[type(item) for item in first]}."
                )
            if issubclass(self._item_type, (list, tuple)):
                self._method = "+"
            elif issubclass(self._item_type, (set, Mapping)):
                self._method = "|"
            else:
                self._method = "none"
            if self._method not in self._methods:
                raise ValueError(
                    f"Unsupported merge method {self._method}. "
                    f"Supported methods are: {list(self._methods.keys())}."
                )
            self._first_call = False
        method = self._method
        for items in zip(*self._iterable):
            yield self._methods[method](items)


if __name__ == "__main__":

    def gen():
        print("Generating...")
        yield {"a": 1}
        print("Generating...")
        yield {"b": 2}

    iterables = [
        # gen(),
        [{"a": 1}, {"b": 2}],
        [{"c": 3}, {"d": 4}],
        # [(1, 2), (3, 4)],
        # [(5, 6), (7, 8)],
    ]

    merger = Merge(MergeConfig())(iterables)
    for item in merger:
        print(item)
    print("------------------------------")
    for item in merger:
        print(item)
