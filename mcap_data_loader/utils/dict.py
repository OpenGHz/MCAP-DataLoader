from typing import Tuple, TypeVar, Any, Dict, Generic, List, Optional, Union
from typing_extensions import Annotated
from collections.abc import Mapping, Iterable, Hashable, Callable
from pydantic import BeforeValidator, PlainSerializer
from cachetools import cached
from collections import defaultdict


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


def _serialize_by_cache(v: Any) -> Dict[K, K]:
    return v.cache


cache_json_serializer = PlainSerializer(_serialize_by_cache, when_used="json")

MappingCall = Annotated[
    Callable[[K], K], BeforeValidator(_validate_mapping_call), cache_json_serializer
]

PredCallType = Callable[[Hashable], Tuple[Hashable, int]]
PredDictType = Dict[Hashable, Tuple[Hashable, int]]
MergeConfigType = Dict[Hashable, List[Hashable]]


def sort_by_priority(values: Iterable, priorities: Iterable) -> list:
    """
    Sorts the values based on their corresponding priorities in ascending order
    (i.e., lower priority numbers indicate higher precedence).
    Elements with equal priorities retain their original relative order (stable sort).

    Args:
        values (Iterable): An iterable of values to be sorted.
        priorities (Iterable): An iterable of priority values corresponding to `values`.
                               Must have the same length as `values`.

    Returns:
        list: A list of values sorted by priority in ascending order,
              preserving original order for items with equal priority.

    Raises:
        ValueError: If `values` and `priorities` have different lengths.
    """
    if len(values) != len(priorities):
        raise ValueError("values 和 priorities 长度必须相同")

    paired = zip(priorities, values)
    sorted_pairs = sorted(paired, key=lambda x: x[0])
    return [value for priority, value in sorted_pairs]


def create_merge_config(
    keys: Iterable[Hashable], pred: PredCallType
) -> MergeConfigType:
    ori_key_dict = defaultdict(list)
    priority_dict = defaultdict(list)
    for k in keys:
        new_k, prior = pred(k)
        ori_key_dict[new_k].append(k)
        priority_dict[new_k].append(prior)
    config = {}
    for new_key, ori_keys in ori_key_dict.items():
        priorities = priority_dict[new_key]
        sorted_ori_keys = sort_by_priority(ori_keys, priorities)
        config[new_key] = sorted_ori_keys
    return config


def merge_values_with_config(
    d: dict, config: MergeConfigType, method: Callable[[Iterable], Any], *args, **kwargs
) -> dict:
    """Merge values in a dictionary based on a merge configuration.

    Args:
        d: The input dictionary with hashable keys.
        config: A merge configuration dictionary mapping new keys to lists of original keys.
        method: A function to merge the list of values for each new_key. Default is sum.
    Returns:
        A new dictionary with merged values.
    """
    return {
        new_key: (
            method((d[k] for k in ori_keys), *args, **kwargs)
            if len(ori_keys) > 1
            else d[ori_keys[0]]
        )
        for new_key, ori_keys in config.items()
    }


def merge_values_with_pred(
    d: dict, pred: PredCallType, method: Callable[[Iterable], Any], *args, **kwargs
) -> dict:
    """Merge values in a dictionary based on a predicate function.

    Args:
        d: The input dictionary with hashable keys.
        pred: A predicate function that takes a key and returns a tuple of (new_key, priority).
            The values with the same new_key will be merged based on their priority, i.e.
            lower priority values will be considered first.
        method: A function to merge the list of values for each new_key. Default is sum.
    Returns:
        A new dictionary with merged values.
    """
    value_dict = defaultdict(list)
    priority_dict = defaultdict(list)
    for k, v in d.items():
        new_k, prior = pred(k)
        # print(f"{k} -> {new_k} with priority {prior}")
        value_dict[new_k].append(v)
        priority_dict[new_k].append(prior)
    merged_dict = {}
    for key, values in value_dict.items():
        priorities = priority_dict[key]
        if len(priorities) == 1:
            merged_dict[key] = values[0]
        else:
            sorted_values = sort_by_priority(values, priorities)
            merged_dict[key] = method(sorted_values, *args, **kwargs)
    return merged_dict


class MergeValuesCall:
    """A callable class to merge dictionary values based on a predicate or configuration."""

    def __init__(
        self,
        method: Callable[[Iterable], Any],
        pred: Optional[Union[PredCallType, PredDictType]] = None,
        config: Optional[MergeConfigType] = None,
        *args,
        **kwargs,
    ):
        # TODO: use a single config class as input
        if not callable(pred):
            pred = CallableKeyMappingDict(pred)
        self.method = method
        self.pred = pred
        self.config = config
        self.args = args
        self.kwargs = kwargs
        if None not in (pred, config):
            raise ValueError("Only one of pred or config can be provided.")
        elif (pred, config) == (None, None):
            raise ValueError("One of pred or config must be provided.")

    def __call__(self, d: dict) -> dict:
        if self.config is None:
            self.config = create_merge_config(d.keys(), self.pred)
        return merge_values_with_config(
            d, self.config, self.method, *self.args, **self.kwargs
        )

    @property
    def cache(self) -> Dict:
        return self.config


def pass_through(data):
    return data


def _validate_merge_values_call(v):
    if not callable(v):
        if v:
            return MergeValuesCall(**v)
        return pass_through
    return v


MergeValuesCallType = Annotated[
    Callable[[dict], dict],
    BeforeValidator(_validate_merge_values_call),
    cache_json_serializer,
]


class PredReplaceTo:
    """A predicate callable that replaces target substrings in keys with specified alternatives."""

    def __init__(
        self,
        targets: List[str],
        sources: List[List[str]],
        to_targets: Optional[List[str]] = None,
    ):
        if len(targets) != len(set(targets)):
            raise ValueError(f"targets must be unique: {targets}")
        self.targets = targets
        self.sources = sources
        self._replace_target = to_targets is not None
        if self._replace_target:
            if len(to_targets) != len(targets):
                raise ValueError("to_targets length must match targets length")
        self.to_targets = to_targets or targets

    def __call__(self, k: str) -> Tuple[str, int]:
        to_targets = self.to_targets
        for i, target in enumerate(self.targets):
            if target in k:
                if self._replace_target:
                    k = k.replace(target, to_targets[i])
                return (k, 0)
        for target, source_list in zip(self.to_targets, self.sources):
            for prior, source in enumerate(source_list):
                if source in k:
                    return (k.replace(source, target), prior + 1)
        return (k, 0)


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

    # from pprint import pprint
    # from pydantic import BaseModel

    # class TestMappingDictModel(BaseModel):
    #     mapping: MappingCall[str]

    # test_dict = {"a": "alpha", "b": "beta"}
    # model_instance = TestMappingDictModel(mapping=test_dict)
    # pprint(model_instance.model_dump(mode="json"))
    # model_instance = TestMappingDictModel(mapping=lambda x: x.upper())
    # model_instance.mapping("gamma")
    # pprint(model_instance.model_dump(mode="json"))

    data = {
        "b": [20],
        "a": [10],
        "d": [40],
        "c": [30],
        "e": 0,
    }

    # def predicate(key):
    #     if key in ["a", "b"]:
    #         return ("group1", key)  # group1
    #     elif key in ["c", "d"]:
    #         return ("group2", 0)  # group2
    #     else:
    #         return (key, 0)  # keep original key

    predicate = PredReplaceTo(
        ["group1", "group2"],
        [["a", "b"], ["d", "c"]],
    )

    merged = merge_values_with_pred(data, predicate, method=sum, start=[])

    def assert_merged(merged):
        assert merged["group1"] == [10, 20], merged
        assert merged["group2"] == [40, 30], merged
        assert merged["e"] == 0, merged

    assert_merged(merged)

    config = create_merge_config(data.keys(), predicate)
    from pprint import pprint

    print("Merge Config:")
    pprint(config)

    merged_with_config = merge_values_with_config(data, config, method=sum, start=[])
    assert_merged(merged_with_config)

    print("Merged Result:")
    pprint(merged_with_config)

    merge_call = MergeValuesCall(method=sum, pred=predicate, start=[])
    merged_with_call = merge_call(data)
    assert_merged(merged_with_call)
