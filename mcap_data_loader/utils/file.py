from typing import Optional
from collections.abc import Generator
from pathlib import Path


def find_file_paths(
    root: Path, name: str, max_depth: Optional[int] = None
) -> Generator[Path]:
    """
    Performs a depth-first search under the given directory to find files with the specified name,
    with an optional maximum search depth.

    Args:
        root (Path): The root directory to start the search from.
        name (str): The filename to search for.
        max_depth (Optional[int]): The maximum depth to traverse into subdirectories.
                                   If None, there is no depth limit.

    Yields:
        Generator[Path]: Paths to all files matching the given name.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"The specified path is not a directory: {root}")

    def _walk_with_depth(current_path: Path, current_depth: int):
        if max_depth is not None and current_depth > max_depth:
            return
        try:
            for item in current_path.iterdir():
                if item.is_dir():
                    yield from _walk_with_depth(item, current_depth + 1)
                elif item.is_file() and item.name == name:
                    yield item
        except PermissionError as e:
            print(e)

    yield from _walk_with_depth(root, 0)
