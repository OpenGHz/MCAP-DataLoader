from pydantic import BaseModel
from typing import Tuple, Union
from collections.abc import Generator, Iterable, Mapping
from mcap_data_loader.pipelines.basis import Pipe, T


class NestedZipConfig(BaseModel, frozen=True):
    """Configuration for NestedZip pipeline."""

    depth: int = -1
    """Depth of nesting for zipping. A depth of 0 means no zipping, 
    1 means a single level of zipping, and so on. If depth < 0, zipping continues
    until the base_type or non-iterable is encountered. If fixed is True, the depth will be calculated
    based on the base_type on the first iteration."""
    base_type: Union[type, Tuple[type]] = (Mapping, str, bytes)
    """Base type(s) to stop zipping when depth is negative."""
    fixed: bool = True
    """Whether the depth is fixed or determined dynamically based on the base_type.
    The latter will be slower, but more flexible."""

    def model_post_init(self, context):
        if not self.fixed and self.depth > 0:
            raise ValueError("If fixed is False, depth must be negative.")


class NestedZip(Pipe[Tuple[T, ...]]):
    def __init__(self, config: NestedZipConfig):
        self.config = config
        self._first_iter = True
        self._depth = config.depth

    def __iter__(self) -> Generator[Tuple[T, ...]]:
        if self._first_iter:
            self._first_iter = False
            if self._depth < 0:
                if self.config.fixed:
                    self._depth = self._get_depth()
        if self._depth < 0:
            raise NotImplementedError(
                "Dynamic depth determination is only supported for fixed=True."
            )
        return self._recursive_iter(self._iterable, self._depth)

    def _get_depth(self) -> int:
        depth = 0
        base_type = self.config.base_type
        current = self._iterable
        while True:
            current = next(iter(current))
            if isinstance(current, base_type) or not isinstance(current, Iterable):
                break
            depth += 1
        return depth

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
    expected_results[-1] = expected_results[2]

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
