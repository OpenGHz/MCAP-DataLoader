from typing import Union, Dict, List, Set, TypeVar
from typing_extensions import Annotated
from collections.abc import Iterable, Iterator, Mapping, Hashable
from pydantic import PlainValidator, AfterValidator, validate_call
from functools import wraps


T = TypeVar("T")


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


NonIteratorIterable = Annotated[
    Iterable[T],
    PlainValidator(validate_iterable_not_iterator),
]
ConstrainedIterable = Annotated[Iterable[T], PlainValidator(validate_iterable)]
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
