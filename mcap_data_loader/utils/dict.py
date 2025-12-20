from typing import Tuple, TypeVar, Callable, Any, Dict, Generic
from typing_extensions import Annotated
from collections.abc import Mapping, Iterable, Hashable
from pydantic import BeforeValidator, PlainSerializer
from cachetools import cached


T = TypeVar("T")


def iterable2dict(iterable: Iterable[T]) -> Mapping[int, T]:
    """Convert an iterable to a dictionary with integer keys."""
    if isinstance(iterable, Mapping):
        return iterable
    return {i: item for i, item in enumerate(iterable)}


def dict2tuple(d: Mapping[int, T]) -> Tuple[T, ...]:
    """Convert a dictionary with integer keys to an iterable."""
    return tuple(d[i] for i in range(len(d)))


def dict2tuple_sort(d: Mapping[Any, T]) -> Tuple[T, ...]:
    """Convert a dictionary with float keys to an iterable."""
    return tuple(d[k] for k in sorted(d.keys()))


def valmap_depth(func: Callable, d: dict, depth: int = -1):
    """
    Recursively apply `func` to the values of a dictionary, up to a specified depth.

    Args:
        func: A callable applied to non-dict values.
        d: Input dictionary.
        depth: Maximum recursion depth.
            - If depth >= 0: apply `func` to values at levels <= depth.
            - If depth < 0: recurse infinitely (i.e., until values are no longer dicts).
    """
    if depth < 0:
        # Infinite recursion: keep recursing into dicts
        return {
            k: valmap_depth(func, v) if isinstance(v, dict) else func(v)
            for k, v in d.items()
        }
    else:
        # Limited recursion
        return {
            k: valmap_depth(func, v, depth - 1) if depth > 0 else func(v)
            for k, v in d.items()
        }


def update_if(
    target: dict,
    source: dict,
    func: Callable[..., bool] = bool,
    strict: bool = False,
    intersection: bool = False,
):
    """Update `target` dictionary with `source` dictionary in a conditional manner.
    Args:
        target: The dictionary to be updated.
        source: The dictionary from which to copy key-value pairs.
        func: A callable that takes a value and returns a boolean.
            Default is `bool`, which means values that are truthy will be updated.
        strict: If False, directly update keys from source that are not in target without checking.
        intersection: If True, only consider keys that are already present in `target`.
    """
    for k, v in source.items():
        k_n_i = k not in target
        if intersection and k_n_i:
            continue
        if ((not strict) and k_n_i) or func(v):
            target[k] = v


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class CallableDict(Dict[K, V]):
    """A dictionary that allows access to its values using a callable interface."""

    def __call__(self, key: K, default: V = None) -> V:
        return self.get(key, default)


class CallableKeyMappingDict(Dict[K, K], Generic[K]):
    """A dictionary that returns the key itself if the key is not found."""

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            if callable(args[0]):
                return
        super().__init__(*args, **kwargs)

    def __call__(self, key: K) -> K:
        return self.get(key, key)

    @property
    def cache(self) -> Dict[K, K]:
        return self


def _validate_mapping_call(v: Any):
    if not callable(v):
        return CallableKeyMappingDict(v)
    return v if hasattr(v, "cache") else cached(cache={})(v)


def _serialize_mapping_call(v: Any) -> Dict[K, K]:
    return v.cache


MappingCall = Annotated[
    Callable[[K], K],
    BeforeValidator(_validate_mapping_call),
    PlainSerializer(_serialize_mapping_call, when_used="json"),
]


if __name__ == "__main__":
    # print("Testing valmap_depth function:")
    # complex_dict = {
    #     "a": 1,
    #     "b": {"b1": 2, "b2": {"b21": 3}},
    #     "c": {"c1": 4, "c2": 5},
    # }

    # print("Original dictionary:")
    # print(complex_dict)

    # print("\nApply valmap_depth with depth=-1 (increment all values):")
    # result_depth_neg1 = valmap_depth(lambda x: x + 10, complex_dict, depth=-1)
    # print(result_depth_neg1)

    # simple_dict = {
    #     "0": {
    #         "0.0": 1,
    #         "0.1": 2,
    #     },
    #     "1": {
    #         "1.0": 3,
    #         "1.1": 4,
    #     },
    # }

    # print("\nApply valmap_depth with depth=1 (increment top-level values):")
    # result_depth_1 = valmap_depth(lambda x: x | {"extra": None}, simple_dict, depth=0)
    # print(result_depth_1)

    # print("\nApply valmap_depth with depth=2 (increment up to second-level values):")
    # result_depth_2 = valmap_depth(lambda x: x + 10, simple_dict, depth=1)
    # print(result_depth_2)

    from pprint import pprint
    from pydantic import BaseModel

    class TestMappingDictModel(BaseModel):
        mapping: MappingCall[str]

    test_dict = {"a": "alpha", "b": "beta"}
    model_instance = TestMappingDictModel(mapping=test_dict)
    pprint(model_instance.model_dump(mode="json"))
    model_instance = TestMappingDictModel(mapping=lambda x: x.upper())
    model_instance.mapping("gamma")
    pprint(model_instance.model_dump(mode="json"))
