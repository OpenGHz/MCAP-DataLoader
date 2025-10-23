from typing import List, Union, Dict, TypeVar, Generic
from typing_extensions import Annotated, TypedDict
from enum import Enum
from pathlib import Path
from collections.abc import Iterable, Iterator
from pydantic import PlainValidator
import hashlib
import time
import sys


T = TypeVar("T")


def validate_iterable_not_iterator(value: Iterable) -> Iterable:
    if not isinstance(value, Iterable):
        raise ValueError("Value must be an Iterable")
    if isinstance(value, Iterator):
        raise ValueError("Value must not be an Iterator")
    return value


NonIteratorIterable = Annotated[
    Iterable[T],
    PlainValidator(validate_iterable_not_iterator),
]


SlicesType = Union[List[tuple], tuple, int]
DictableSlicesType = Union[Dict[str, SlicesType], SlicesType]
DictableIndexesType = Union[Dict[str, List[int]], List[int]]


class DataStamped(TypedDict, Generic[T]):
    t: int
    data: T


DictDataStamped = Dict[str, DataStamped[T]]


if sys.version_info >= (3, 10):
    from functools import partial

    zip = partial(zip, strict=True)
else:
    from more_itertools import zip_equal as zip  # noqa: F401


def get_stamp_ms() -> int:
    return int(time.time() * 1e3)


class bcolors:
    MAGENTA = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ReprEnum(Enum):
    """
    Only changes the repr(), leaving str() and format() to the mixed-in type.
    """


class StrEnum(str, ReprEnum):
    """
    Enum where members are also (and must be) strings
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError(f"too many arguments for str(): {values!r}")
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError(f"{values[0]!r} is not a string")
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError(f"encoding must be a string, not {values[1]!r}")
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()

    def __str__(self):
        return self.value


def multi_slices_to_indexes(slices: SlicesType) -> List[int]:
    """Convert slices to a list of indexes.
    Args:
        slices: can be a int number to use the first n episodes
        or a tuple of (start, end) to use the episodes from start to
        end (not included the end), e.g. (50, 100) or a tuple of
        (start, end, suffix) to use the episodes from start to end with the suffix,
        e.g. (50, 100, "augmented") or a list (not tuple!) of
        multi tuples e.g. [(0, 50), (100, 200)].
        Empty slices will be ignored.
    Returns:
        A list of indexes, e.g. [0, 1, ...,] or ['0_suffix', '1_suffix', ...]
    Raises:
        ValueError: if slices is not a tuple or list of tuples
    Examples:
        multi_slices_to_indexes(10) -> [0, 1, 2, ..., 9]
        multi_slices_to_indexes((5, 10)) -> [5, 6, 7, 8, 9]
        multi_slices_to_indexes((5, 7, "_suffix")) -> ['5_suffix', '6_suffix', '7_suffix']
        multi_slices_to_indexes([(1, 4), (8, 10)]) -> [1, 2, 3, 8, 9]
    """

    def process_tuple(tuple_slices: tuple) -> list:
        tuple_len = len(tuple_slices)
        if tuple_len == 2:
            start, end = tuple_slices
            suffix = None
        elif tuple_len == 3:
            start, end, suffix = tuple_slices
        elif tuple_len == 0:
            return []
        else:
            raise ValueError(f"tuple_slices length is {tuple_len}, not in ")
        tuple_slices = list(range(start, end))
        if suffix is not None:
            for index, ep in enumerate(tuple_slices):
                tuple_slices[index] = f"{ep}{suffix}"
        return tuple_slices

    if isinstance(slices, int):
        slices = (0, slices)

    if isinstance(slices, tuple):
        slices = process_tuple(slices)
    elif isinstance(slices, list):
        for index, element in enumerate(slices):
            if isinstance(element, int):
                element = (element, element + 1)
            slices[index] = process_tuple(element)
        # flatten the list
        flattened = []
        for sublist in slices:
            flattened.extend(sublist)
        slices = flattened
    else:
        raise ValueError("slices should be tuple or list of tuples")
    return slices


def get_items_by_ext(directory: Union[str, Path], extension: str) -> List[Path]:
    """Get all files or directories in a directory with a specific extension (suffix).
    Args:
        directory (str): The directory to search in.
        extension (str): The file extension to filter by. If empty, return directories.
            If extension is ".", return all files.
        with_directory (bool, optional): Whether to include the directory path in the
            returned file names. Defaults to False.
    Returns:
        List[str]: A list of file or directory names that match the extension.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    if not directory.exists():
        return []
    entries = directory.iterdir()
    if extension == ".":
        return [entry for entry in entries if entry.is_file()]
    elif not extension:
        return [entry for entry in entries if entry.is_dir()]
    else:
        return [
            entry
            for entry in entries
            if entry.is_file() and entry.suffix.endswith(extension)
        ]


def file_hash(
    file_path: Union[str, Path], algorithm: str = "md5", chunk_size: int = 1024**3
) -> str:
    """Compute the hash of a file using the specified algorithm.
    Args:
        filepath (Union[str, Path]): Path to the file.
        algorithm (str, optional): Hash algorithm to use. Defaults to "md5".
        chunk_size (int, optional): Size of chunks to read the file. Defaults to 1GB.
    Returns:
        str: Hexadecimal hash string.
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def get_fully_qualified_class_name(obj_or_cls):
    if isinstance(obj_or_cls, type):
        cls = obj_or_cls
    else:
        cls = type(obj_or_cls)
    return f"{cls.__module__}.{cls.__qualname__}"


if __name__ == "__main__":
    # assert multi_slices_to_indexes(()) == []
    # assert multi_slices_to_indexes(10) == list(range(10))
    # assert multi_slices_to_indexes((5, 10)) == list(range(5, 10))
    # assert multi_slices_to_indexes((5, 10, "suffix")) == [
    #     f"{i}suffix" for i in range(5, 10)
    # ]
    # assert multi_slices_to_indexes([(1, 4), (8, 10)]) == list(range(1, 4)) + list(
    #     range(8, 10)
    # )

    print(get_items_by_ext("data/example", ".mcap"))
    print(get_items_by_ext("data/example", ""))
    print(get_items_by_ext("data/example", "."))
