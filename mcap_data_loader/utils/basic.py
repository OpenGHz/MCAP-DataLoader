from typing import (
    List,
    Union,
    Dict,
    TypeVar,
    Generic,
    Any,
    Type,
    Optional,
    Protocol,
    Hashable,
    get_origin,
    get_args,
)
from typing_extensions import Annotated, TypedDict, runtime_checkable
from enum import Enum
from pathlib import Path
from collections.abc import Iterable, Iterator, Callable
from pydantic import PlainValidator, validate_call
from functools import wraps
from inspect import isclass
import hashlib
import time
import sys
import importlib


def validate_call_once(func):
    validated_func = validate_call(func)
    called = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            return validated_func(*args, **kwargs)
        else:
            # 后续调用直接使用原始函数，不验证
            return func(*args, **kwargs)

    return wrapper


def validate_iterable_not_iterator(value: Iterable) -> Iterable:
    if not isinstance(value, Iterable):
        raise ValueError("Value must be an Iterable")
    if isinstance(value, Iterator):
        raise ValueError("Value must not be an Iterator")
    return value


@runtime_checkable
class DataClassProto(Protocol):
    """Protocol for dataclass types."""

    @classmethod
    def __dataclass_fields__(cls) -> Dict[str, Any]: ...


T = TypeVar("T")
NonIteratorIterable = Annotated[
    Iterable[T],
    PlainValidator(validate_iterable_not_iterator),
]
ReturnT = TypeVar("ReturnT")
KeyT = TypeVar("KeyT", bound=Hashable)
DataT = TypeVar("DataT")


SlicesType = Union[List[tuple], tuple, int]
DictableSlicesType = Union[Dict[str, SlicesType], SlicesType]
DictableIndexesType = Union[Dict[str, List[int]], List[int]]


class DataStamped(TypedDict, Generic[T]):
    t: int
    data: T

    @staticmethod
    def map_dict(
        data: Dict[KeyT, "DataStamped[DataT]"],
        func: Callable[[DataT], ReturnT],
        keys: Optional[Iterable[KeyT]] = None,
    ) -> Dict[KeyT, "DataStamped[ReturnT]"]:
        result = {}
        keys = data.keys() if keys is None else keys
        for key in keys:
            stamped = data[key]
            result[key] = {
                "t": stamped["t"],
                "data": func(stamped["data"]),
            }
        return result


DictDataStamped = Dict[str, DataStamped[T]]


if sys.version_info >= (3, 10):
    from functools import partial

    zip = partial(zip, strict=True)
else:
    from more_itertools import zip_equal as zip  # noqa: F401


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


class Rate:
    def __init__(self, rate_hz: float):
        self._interval = 1.0 / rate_hz
        self._last_time = time.perf_counter()

    def sleep(self):
        now = time.perf_counter()
        elapsed = now - self._last_time
        sleep_time = self._interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.perf_counter()

    def reset(self):
        self._last_time = time.perf_counter()


class InputSleeper:
    def reset(self):
        pass

    def sleep(self):
        input("Press Enter to continue...")


def create_sleeper(rate_hz: float) -> Union[Rate, InputSleeper]:
    if rate_hz == 0:
        return InputSleeper()
    else:
        return Rate(rate_hz)


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
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
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


def float_range(start: float, stop: float, step: int = 1):
    """
    Generates a sequence of floating-point numbers from start (inclusive) to stop (exclusive),
    with a step size determined by `step` (1 means 0.1, 2 means 0.2, etc.).
    Requires that `start` and `stop` share the same "prefix" (i.e., floor(start * 10) == floor(stop * 10));
    otherwise, raises a ValueError.

    Args:
        start (float): The starting value.
        stop (float): The ending value (not included).
        step (int): Step size in units of 0.1 (default is 1).

    Examples:
        float_range(1.0, 1.5, 1) -> [1.0, 1.1, 1.2, 1.3, 1.4]
        float_range(1.0, 1.5, 2) -> [1.0, 1.2, 1.4]
        float_range(1.2, 2.1) -> ValueError
    """
    if step <= 0:
        raise ValueError("Step must be a positive integer.")

    # Convert input to "tenths" (integer representation scaled by 10)
    def to_tenth(x: float) -> int:
        tenth = int(x * 10)
        if abs(x * 10 - tenth) > 1e-9:
            raise ValueError(f"Input {x} has more than one decimal place.")
        return tenth

    start_tenth = to_tenth(start)
    stop_tenth = to_tenth(stop)

    # Check if both values lie within the same "tenths decade" (i.e., same prefix)
    if start_tenth // 10 != stop_tenth // 10:
        raise ValueError(
            f"Start ({start}) and stop ({stop}) have inconsistent prefixes."
        )

    result = []
    current = start_tenth
    while current < stop_tenth:
        value = current / 10.0
        result.append(round(value, 1))
        current += step

    return result


def get_full_class_name(obj: Union[Any, Type]) -> str:
    cls = obj if isclass(obj) else obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def get_class_type(class_path: str) -> Type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def remove_util(string: str, stop: str, include_stop: bool = True) -> str:
    """Remove part of the string before the stop string (including or excluding the stop string).
    Args:
        string (str): The original string.
        stop (str): The stop string.
        include_stop (bool, optional): Whether to include the stop string in the result. Defaults to True.
    Returns:
        str: The modified string.
    Raises:
        ValueError: if stop string is empty.
    Examples:
        remove_util("123.abc", ".", False) -> "abc"
    """
    if not stop:
        raise ValueError("stop string cannot be empty")
    index = string.find(stop)
    bias = 0 if include_stop else len(stop)
    result = string[index + bias :] if index != -1 else string
    return result


def resolve_generic_type(cls: Type, target_origin: Type) -> Optional[Type]:
    """
    Recursively resolves the concrete type argument corresponding to `target_origin`
    in the generic base classes of `cls`.

    Handles multi-level generic inheritance. For example:
        class A(Generic[T]): ...
        class B(A[int]): ...
        class C(B): ...

    In this case, calling `resolve_generic_type(C, A)` returns `int`.
    """
    if not hasattr(cls, "__orig_bases__"):
        return None

    for base in cls.__orig_bases__:
        origin = get_origin(base)
        args = get_args(base)
        # Direct match with the target base class, e.g., Basis[str]
        if origin is target_origin:
            return args[0] if args else None
        # If the base itself is a generic class (e.g., CustomBasis[dict])
        if isinstance(origin, type) and issubclass(origin, Generic):
            inner_type = resolve_generic_type(origin, target_origin)
            if isinstance(inner_type, TypeVar):
                # If the resolved type is a TypeVar (e.g., T), perform substitution
                type_params = getattr(origin, "__parameters__", ())
                mapping = dict(zip(type_params, args))
                return mapping.get(inner_type, inner_type)
            elif inner_type is not None:
                return inner_type
    return None


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

    # print(get_items_by_ext("data/example", ".mcap"))
    # print(get_items_by_ext("data/example", ""))
    # print(get_items_by_ext("data/example", "."))

    # print(float_range(1.0, 1.5))  # Default step = 0.1: [1.0, 1.1, 1.2, 1.3, 1.4]
    # print(float_range(1.0, 1.5, 2))  # Step = 0.2: [1.0, 1.2, 1.4]
    # print(float_range(1.0, 1.6, 3))  # Step = 0.3: [1.0, 1.3] (1.6 is excluded)
    # print(float_range(1.0, 1.62, 3))  # Step = 0.3: [1.0, 1.3, 1.6] (1.62 is truncated to 1.6; 1.6 is excluded)
    # print(float_range(1.0, 2.1))         # ValueError: prefix 1 vs 2 mismatch
    # print(float_range(1.0, 1.5, -1))     # ValueError: Step must be a positive integer.

    # result = remove_util("123.abc", ".", False)
    # assert result == "abc", result
    # result = remove_util("123.abc", ".", True)
    # assert result == ".abc", result
    # result = remove_util("123abc", "123", False)
    # assert result == "abc", result
    # result = remove_util("12ab34", "ab")
    # assert result == "ab34", result
    # assert remove_util("12ab34", "567") == "12ab34"

    import numpy as np
    import time

    data = {
        "a": {"t": 1, "data": [1, 2]},
        "b": {"t": 2, "data": [3, 4]},
    }
    start = time.perf_counter()
    result = DataStamped.map_dict(data, np.array)
    print("Time taken:", time.perf_counter() - start)
    for value in result.values():
        print(value["t"])
        print(value["data"].shape)
