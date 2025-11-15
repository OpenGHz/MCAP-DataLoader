from pydantic import BaseModel
from typing import Tuple
from collections.abc import Generator
from mcap_data_loader.pipelines.basis import Pipeline, T


class NestedZipConfig(BaseModel, frozen=True):
    """Configuration for NestedZip pipeline."""

    depth: int = 1
    """Depth of nesting for zipping. A depth of 0 means no zipping, 
    1 means a single level of zipping, and so on."""


class NestedZip(Pipeline[Tuple[T, ...]]):
    def __init__(self, config: NestedZipConfig):
        self.config = config

    def __iter__(self) -> Generator[Tuple[T, ...]]:
        return self._recursive_iter(self._iterable, self.config.depth)

    def _recursive_iter(self, iterables, depth) -> Generator[Tuple[T, ...]]:
        if depth > 0:
            for items in zip(*iterables):
                yield from self._recursive_iter(items, depth - 1)
        else:
            yield iterables


if __name__ == "__main__":
    from pprint import pprint

    iterables = [
        [[1, 2], [3, 4]],
        [["a", "b"], ["c", "d"]],
    ]

    expected_results = {
        0: [iterables],
        1: tuple(zip(*iterables)),
        2: (
            (1, "a"),
            (2, "b"),
            (3, "c"),
            (4, "d"),
        ),
    }

    # pprint(expected_results)

    for depth, expected in expected_results.items():
        print(f"Depth: {depth}.")
        nested = NestedZip(NestedZipConfig(depth=depth))(iterables)
        for i, item in enumerate(nested):
            pprint(item)
            assert item == expected[i], (
                f"Depth {depth}, Item {i} failed: {item} != {expected[i]}"
            )
    print("All tests passed.")
