"""Basic utility functions and classes for MCAP Data Loader.TODO: simplify and split into multiple files."""

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
    Literal,
    get_origin,
    get_args,
)
from collections.abc import Callable
from typing_extensions import runtime_checkable
from pathlib import Path
from pydantic import BaseModel, ConfigDict, ImportString, validate_call
from functools import wraps
from inspect import isclass
from logging import getLogger
from contextlib import suppress
import hashlib
import operator
import time
import sys
import json


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
T = TypeVar("T")


# The configuration framework moved to the standalone `cfgable` package.
# These names are re-exported so existing
# `from mcap_data_loader.utils.basic import ...` imports keep working.
from cfgable import (  # noqa: F401
    validate_field,
    ForceSetAttr,
    force_set_attr,
    force_validate_field,
    DataClassProto,
    import_string,
    get_fully_qualified_class_name,
)


SlicesType = Union[List[tuple], tuple, int]
DictableSlicesType = Union[Dict[str, SlicesType], SlicesType]
DictableIndexesType = Union[Dict[str, List[int]], List[int]]


if sys.version_info >= (3, 10):
    from functools import partial

    zip = partial(zip, strict=True)
else:
    from more_itertools import zip_equal as zip  # noqa: F401


class DataBasicConfig(BaseModel, frozen=True):
    """Basic configuration for data processing."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    device: Optional[str] = None
    """Device to use for data processing, e.g., 'cpu' or 'cuda'."""
    dtype: Optional[Union[Literal["auto"], str]] = None
    """Data type to use for data processing, e.g., 'float32' or 'int64'."""


class Rate:
    def __init__(self, rate_hz: float):
        """Initialize the Rate object with the desired frequency in Hertz.
        Args:
            rate_hz (float): The frequency in Hertz at which to run.
                If set to negative, no sleeping will occur.
        Raises:
            DivisionByZeroError: If rate_hz is zero.
        """
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
        getLogger(self.__class__.__name__).info("Press Enter to continue...")
        return input()


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
        extension (str): The file extension to filter by. If empty, return all directories.
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


# get_fully_qualified_class_name moved to `cfgable` (re-exported above).


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


# import_string moved to `cfgable` (re-exported above).


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


def has_nested_class_strict(cls: Type) -> bool:
    for name, obj in cls.__dict__.items():
        if (
            isclass(obj)
            and obj.__module__ == cls.__module__  # 同一模块
            and obj.__qualname__.startswith(cls.__qualname__ + ".")
        ):
            return True
    return False


def not_implemented(func):
    """Decorator that makes a function raise NotImplementedError when called."""

    func.__isnotimplemented__ = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{func.__qualname__} is not implemented")

    return wrapper


def is_not_implemented(func) -> bool:
    """Check if a function is decorated with @not_implemented."""
    return getattr(func, "__isnotimplemented__", False)


def try_to_get_attr(obj: Any, attrs: List[str], default: Any = object) -> Any:
    """Try to get nested attributes from an object.
    Args:
        obj (Any): The object to get attributes from.
        attrs (List[str]): The list of attribute names to get.
        default (Any, optional): The default value to return if any attribute is not found. Defaults to None.
    Returns:
        Any: The value of the nested attribute or the default value.
    Raises:
        AttributeError: if none of the attributes are found and default is not provided.
    """
    for attr in attrs:
        with suppress(AttributeError):
            return operator.attrgetter(attr)(obj)
    if default is not object:
        return default
    raise AttributeError(f"None of the attributes {attrs} found in {obj}.")


def cfgize(func: Callable) -> Callable:
    """Decorator to convert a callable into one that accepts a config and additional arguments."""

    @wraps(func)
    def wrapper(config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        return func(*args, **(config or {}), **kwargs)

    return wrapper


def is_cached(func: Callable) -> bool:
    return hasattr(func, "cache_info") and hasattr(func, "cache_clear")


def save_current_command(
    json_path: str, key: str, as_list: bool = False
) -> Union[str, List[str]]:
    """
    Save the command used to execute the current script to a specified JSON file under a given key.
    Args:
        json_path (str): Path to the target JSON file (directories will be created if needed)
        key (str): The key in the JSON file under which to store the command
        as_list (bool):
            - If True, save as a list of command components (e.g., ["python", "script.py", "--arg", "val"])
            - If False (default), save as a single string (e.g., "python script.py --arg val")
    Returns:
        Union[str, List[str]]: The command representation that was saved
    """
    command = [sys.executable] + sys.argv
    command_repr = command if as_list else " ".join(command)

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(json_path).exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = command_repr

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # print(f"[INFO] Saved current command to {json_path} under key '{key}'")
    return command_repr


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

    # class A(Generic[T]):
    #     class B(Generic[T]):
    #         pass

    # class C:
    #     pass

    # assert has_nested_class_strict(A)
    # assert not has_nested_class_strict(C)
    # print("All tests passed.")

    # def sample_not_implemented():
    #     @not_implemented
    #     def func():
    #         pass

    #     try:
    #         func()
    #     except NotImplementedError:
    #         print("NotImplementedError raised as expected.")
    #     else:
    #         print("Error: NotImplementedError was not raised.")

    #     assert is_not_implemented(func)

    # sample_not_implemented()

    # class a:
    #     class b:
    #         class c:
    #             pass

    # assert get_full_class_name(a.b.c) == "__main__.a.b.c"
    # assert try_to_get_attr(a, ["b.d", "b.c"]) is a.b.c

    # save_current_command("test_command.json", "last_command", as_list=False)
    pass
