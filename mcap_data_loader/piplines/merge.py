from pydantic import BaseModel
from collections.abc import Mapping, Generator
from collections import ChainMap
from mcap_data_loader.piplines.basis import Pipeline, T


class MergeConfig(BaseModel):
    method: str = "auto"


class Merge(Pipeline[T]):
    def __init__(self, config: MergeConfig) -> None:
        self.config = config
        self._methods = {
            "ChainMap": lambda items: ChainMap(*items),
            "+": self._sum,
            "|": self._or,
            "none": lambda items: items,
        }

    def _sum(self, items):
        base = items[0]
        for item in items[1:]:
            base += item
        return base

    def _or(self, items):
        base = items[0]
        for item in items[1:]:
            base |= item
        return base

    def __iter__(self) -> Generator[T]:
        if self.config.method == "auto":
            first = next(zip(*self._iterables))
            item_type = type(first[0])
            if not all(isinstance(item, item_type) for item in first):
                raise ValueError(
                    f"All items in the first iterable must be of type {item_type}, "
                    f"but got {[type(item) for item in first]}."
                )
            if issubclass(item_type, Mapping):
                self.config.method = "ChainMap"
            elif issubclass(item_type, (list, tuple)):
                self.config.method = "+"
            elif issubclass(item_type, set):
                self.config.method = "|"
            else:
                self.config.method = "none"
        if self.config.method not in self._methods:
            raise ValueError(
                f"Unsupported merge method {self.config.method}. "
                f"Supported methods are: {list(self._methods.keys())}."
            )
        for items in zip(*self._iterables):
            yield self._methods[self.config.method](items)


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

    merger = Merge(MergeConfig()).wrap(iterables)
    for item in merger:
        print(item)
    print("------------------------------")
    for item in merger:
        print(item)
