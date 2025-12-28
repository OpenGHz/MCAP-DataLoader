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
    Set,
    Literal,
    get_origin,
    get_args,
)
from typing_extensions import Annotated, TypedDict, runtime_checkable
from enum import Enum
from pathlib import Path
from collections.abc import Iterable, Iterator, Callable, Mapping, Hashable
from pydantic import (
    BaseModel,
    PlainValidator,
    AfterValidator,
    ConfigDict,
    ImportString,
    validate_call,
)
from functools import wraps
from inspect import isclass
from logging import getLogger
from contextlib import suppress
from copy import deepcopy
from statistics import mean
from toolz import curry
from copy import copy
import hashlib
import operator
import time
import sys


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
T = TypeVar("T")


def validate_field(obj: BaseModel, name: str, value: Any):
    """Validate a field value using the Pydantic model's validator."""
    obj.__pydantic_validator__.validate_assignment(obj, name, value)


class ForceSetAttr(Generic[BaseModelT]):
    """Context manager to temporarily allow setting attributes on frozen Pydantic models."""

    def __init__(self, obj: BaseModelT):
        if not isinstance(obj, BaseModel):
            raise TypeError("Only Pydantic BaseModel instances are supported.")
        self._obj = obj

    def __enter__(self) -> BaseModelT:
        self._original_setattr = self._obj.__class__.__setattr__
        self._obj.__class__.__setattr__ = self._setattr
        return self._obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._obj.__class__.__setattr__ = self._original_setattr

    def _setattr(self, name, value):
        obj = self._obj
        config = obj.model_config
        if config.get("frozen", False):
            if config.get("validate_assignment", False):
                validate_field(obj, name, value)
            else:
                object.__setattr__(obj, name, value)
        else:
            setattr(obj, name, value)


def force_set_attr(method):
    """Decorator to force attribute setting on frozen Pydantic models."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with ForceSetAttr(self):
            return method(self, *args, **kwargs)

    return wrapper


force_validate_field = force_set_attr(validate_field)


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


def validate_iterable(value: Iterable, base_types=(str, bytes, Mapping)) -> Iterable:
    if not isinstance(value, Iterable):
        raise ValueError("Value must be an Iterable")
    if isinstance(value, base_types):
        raise ValueError(f"Value must not be of type {base_types}")
    return value


def validate_iterable_not_iterator(
    value: Iterable, base_types=(str, bytes, Mapping)
) -> Iterable:
    if isinstance(value, Iterator):
        raise ValueError("Value must not be an Iterator")
    return validate_iterable(value, base_types)


def _mapping2list(value: Union[Dict, List]) -> List:
    if isinstance(value, Mapping):
        return list(value.values())
    return value


def _mapping2list_sorted(value: Union[Dict, List]) -> List:
    if isinstance(value, Mapping):
        return [value[key] for key in sorted(value.keys())]
    return value


def _mapping2set(value: Union[Dict, Set]) -> Set:
    if isinstance(value, Mapping):
        return set(value.values())
    return value


@runtime_checkable
class DataClassProto(Protocol):
    """Protocol for dataclass types."""

    @classmethod
    def __dataclass_fields__(cls) -> Dict[str, Any]: ...


NonIteratorIterable = Annotated[
    Iterable[T],
    PlainValidator(validate_iterable_not_iterator),
]
ConstrainedIterable = Annotated[Iterable[T], PlainValidator(validate_iterable)]
ReturnT = TypeVar("ReturnT")
KeyT = TypeVar("KeyT", bound=Hashable)
DataT = TypeVar("DataT")

# convert Mapping to List of values with the original order of keys
ListMapping = Annotated[
    Union[List[T], Mapping[Hashable, T]], AfterValidator(_mapping2list)
]
# convert Mapping to List of values with sorted order of keys
ListMappingSorted = Annotated[
    Union[List[T], Mapping[Hashable, T]], AfterValidator(_mapping2list_sorted)
]
SetMapping = Annotated[
    Union[Set[T], Mapping[Hashable, T]], AfterValidator(_mapping2set)
]

SlicesType = Union[List[tuple], tuple, int]
DictableSlicesType = Union[Dict[str, SlicesType], SlicesType]
DictableIndexesType = Union[Dict[str, List[int]], List[int]]


@curry
def sum_auto_start(iterable: Iterable[T]) -> T:
    """Sum the items in the iterable, starting from the first item."""
    iterator = iter(iterable)
    total = copy(next(iterator))
    for item in iterator:
        total += item
    return total


class DataStamped(TypedDict, Generic[T]):
    t: int
    data: T

    @staticmethod
    def map_dict(
        data: Dict[KeyT, "DataStamped[DataT]"],
        func: Callable[[DataT], ReturnT],
        keys: Optional[Iterable[KeyT]] = None,
        output: Optional[dict] = None,
    ) -> Dict[KeyT, "DataStamped[ReturnT]"]:
        result = output if output is not None else {}
        keys = data.keys() if keys is None else keys
        for key in keys:
            stamped = data[key]
            result[key] = {
                "t": stamped["t"],
                "data": func(stamped["data"]),
            }
        return result

    @staticmethod
    def merge(
        values: Iterable["DataStamped[DataT]"],
        d_method: Callable[[List[DataT]], ReturnT] = sum_auto_start,
        t_method: Callable[[List[int]], int] = mean,
    ) -> "DataStamped[ReturnT]":
        time_list = []
        data_list = []
        for item in values:
            time_list.append(item["t"])
            data_list.append(item["data"])
        return {"t": int(t_method(time_list)), "data": d_method(data_list)}

    @staticmethod
    def create(data: T, t: int = 0) -> "DataStamped[T]":
        return {"t": t, "data": data}


DictDataStamped = Dict[str, DataStamped[T]]
map_dict_data_stamped = curry(DataStamped.map_dict)
merge_data_stamped = curry(DataStamped.merge)


def copy_dict_data_stamped(data: DictDataStamped[T], deep: bool = False):
    """Copy a DictDataStamped object.
    Args:
        data (DictDataStamped[T]): The DictDataStamped object to copy.
        deep (bool, optional): Whether to perform a deep copy. Defaults to False.
    Returns:
        DictDataStamped[T]: The copied DictDataStamped object.
    """
    if deep:
        return deepcopy(data)
    else:
        return {key: value.copy() for key, value in data.items()}


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


@validate_call
def import_string(import_path: ImportString[T]) -> T:
    return import_path


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

    # import numpy as np
    # import time

    # data = {
    #     "a": {"t": 1, "data": [1, 2]},
    #     "b": {"t": 2, "data": [3, 4]},
    # }
    # start = time.perf_counter()
    # result = DataStamped.map_dict(data, np.array)
    # print("Time taken:", time.perf_counter() - start)
    # for value in result.values():
    #     print(value["t"])
    #     print(value["data"].shape)

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

    class a:
        class b:
            class c:
                pass

    assert get_full_class_name(a.b.c) == "__main__.a.b.c"
    assert try_to_get_attr(a, ["b.d", "b.c"]) is a.b.c
