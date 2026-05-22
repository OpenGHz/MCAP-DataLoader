from typing import Optional, Literal
from collections.abc import Generator
from pathlib import Path
from send2trash import send2trash
from send2trash.exceptions import TrashPermissionError
import shutil


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


def _permanent_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def remove_path(
    path: Path, mode: Literal["permanent", "trash"] = "permanent", log: bool = False
) -> bool:
    """Remove the data from the given or last saved path.

    When ``mode == "trash"`` the volume may not allow creating its
    ``.Trash-<uid>`` directory (e.g. shared ``/data`` mounts on servers).
    send2trash raises ``TrashPermissionError`` in that case; fall back to a
    permanent removal so the caller still makes progress.
    """
    if path.exists():
        if mode == "permanent":
            _permanent_remove(path)
        else:
            try:
                send2trash(path)
            except TrashPermissionError as exc:
                if log:
                    print(
                        f"send2trash unavailable for {path} ({exc!r}); "
                        "falling back to permanent removal."
                    )
                _permanent_remove(path)
        return True
    else:
        if log:
            print(f"Path to be removed {path} does not exist.")
        return False
