from toolz import curry
from typing import List, Optional, Any, Set, TypeVar, Dict, Tuple, Union
from collections.abc import Callable, Iterable
import operator


DataT = TypeVar("DataT")
ReturnT = TypeVar("ReturnT")


@curry
def add_prefix_suffix(key: str, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{key}{suffix}"


@curry
def reorder_key(
    key: str,
    from_indices: List[int],
    to_indices: List[int],
    ori_sep: str = "/",
    inner_sep: Optional[str] = None,
    outer_sep: Optional[str] = None,
    strict: bool = True,
) -> str:
    """
    Reorder parts of a key (path-like string) based on specified indices.
    Args:
        key (str): The original key to be reordered.
        from_indices (List[int]): The original indices of the parts to be reordered.
        to_indices (List[int]): The new order of the specified parts.
        ori_sep (str): The separator used in the original key to split parts.
        inner_sep (Optional[str]): The separator to use when merging the reordered parts.
                                   If None, defaults to ori_sep.
        outer_sep (Optional[str]): The separator to use when joining all parts back together.
                                   If None, defaults to ori_sep.
        strict (bool): Whether to raise errors on invalid indices or return the original key.
    Returns:
        str: The reordered key with specified parts merged.
    """
    inner_sep = ori_sep if inner_sep is None else inner_sep
    outer_sep = ori_sep if outer_sep is None else outer_sep
    # split and filter empty parts
    parts = [p for p in key.split(ori_sep) if p]
    if not parts:
        if strict:
            raise ValueError(f"key has no parts to reorder: {key=}")
        return key

    n = len(parts)
    reorder_set = set(from_indices)

    def check_indices(indices: List[int]) -> None:
        if not indices:
            raise ValueError("indices cannot be empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"indices cannot have duplicates, got {indices}")
        for idx in indices:
            if not (0 <= idx < n):
                if strict:
                    raise IndexError(f"index {idx} out of range [0, {n - 1}].{key=}")
                return False
        return True

    # check from_indices
    if not check_indices(from_indices):
        return key
    # check to_indices
    if set(to_indices) != reorder_set or len(to_indices) != len(from_indices):
        raise ValueError("`to_indices` must be a permutation of `from_indices`")
    # 确定要合并的范围（包含重排索引之间的字段）
    min_idx, max_idx = min(from_indices), max(from_indices)

    # 1) 按 to_indices 收集需要重排的字段
    merged_items = [parts[i] for i in to_indices]
    # 2) 把范围内非重排的字段追加，保持它们的原始顺序
    merged_items.extend(
        parts[i] for i in range(min_idx, max_idx + 1) if i not in reorder_set
    )
    merged_field = inner_sep.join(merged_items)

    # 组装新路径：范围前保留，插入合并字段，范围后保留
    prefix = parts[:min_idx]
    suffix = parts[max_idx + 1 :]
    new_parts = [*prefix, merged_field, *suffix]

    return outer_sep + outer_sep.join(new_parts)


@curry
def call_if(
    data: DataT,
    pred: Callable[[DataT], bool],
    func: Callable[[DataT], ReturnT],
    func_else: Callable[[DataT], ReturnT] = lambda x: x,
) -> ReturnT:
    """Call `func` on `data` if `pred(data)` is True, else call `func_else` on `data`.
    Args:
        data: Input data.
        pred: Predicate function to evaluate on data.
        func: Function to call if predicate is True.
        func_else: Function to call if predicate is False. Defaults to identity function.
    Returns:
        Result of calling `func` or `func_else` on `data`.
    """
    if pred(data):
        return func(data)
    return func_else(data)


@curry
def call_multi_args(
    data: DataT,
    func: Callable[[DataT], ReturnT],
    args_kwargs: Iterable[Tuple[Tuple[Any, ...], Dict[str, Any]]],
    chain: bool = True,
) -> Union[ReturnT, List[ReturnT]]:
    """Compose multiple function calls with keyword arguments on data.
    Args:
        data: Input data.
        func: Function to be called multiple times.
        args_kwargs: An iterable of tuples, each containing a tuple of positional arguments
                     and a dictionary of keyword arguments for the function.
        chain: If True, the output of each function call is passed as input to the next.
               If False, each function call is independent and returns a list of results.
    Returns:
        Result of the composed function calls.
    """
    if chain:
        for args, kwargs in args_kwargs:
            data = func(data, *args, **kwargs)
        results = data
    else:
        results = []
        for args, kwargs in args_kwargs:
            result = func(data, *args, **kwargs)
            results.append(result)
    return results


@curry
def fragment_matching(
    key: str,
    frag_in: Optional[Set[str]] = None,
    frag_not: Optional[Set[str]] = None,
    startswith: str = "",
    endswith: str = "",
    logic: str = "and",
) -> bool:
    if startswith and not key.startswith(startswith):
        return False
    if endswith and not key.endswith(endswith):
        return False
    frag_in = set(frag_in or set())
    frag_not = set(frag_not or set())
    if frag_in & frag_not:
        raise ValueError("frag_in and frag_not cannot have common elements")
    is_in = not frag_in or any(frag in key for frag in frag_in)
    is_not = not frag_not or all(frag not in key for frag in frag_not)
    return getattr(operator, f"{logic}_")(is_in, is_not)


@curry
def replace(
    key: str,
    olds: Union[str, List[str]],
    news: Union[str, List[str]],
    counts: Union[int, List[int]] = -1,
) -> str:
    olds = [olds] if isinstance(olds, str) else olds
    news = [news] if isinstance(news, str) else news
    if len(olds) != len(news):
        raise ValueError(f"Length of olds ({olds}) and news ({news}) must be the same")
    counts = [counts] if isinstance(counts, int) else counts
    if len(counts) == 1:
        counts = counts * len(olds)
    if len(counts) != len(olds):
        raise ValueError(
            f"Length of counts ({counts}) must be 1 or the same as olds/news ({olds}/{news})"
        )
    for old, new, count in zip(olds, news, counts):
        key = key.replace(old, new, count)
    return key


if __name__ == "__main__":
    result = reorder_key(
        "/left/leader/arm/pose",
        from_indices=[1, 2],
        to_indices=[2, 1],
        inner_sep="_",
    )
    assert result == "/left/arm_leader/pose", result

    result2 = reorder_key(
        "/base/left/leader/arm/hand/pose",
        from_indices=[2, 3, 4],  # leader, arm, hand
        to_indices=[4, 2, 3],  # hand, leader, arm
        inner_sep="-",
    )
    assert result2 == "/base/left/hand-leader-arm/pose", result2

    result3 = reorder_key(
        "/a/b/c/d/e",
        from_indices=[1, 3],
        to_indices=[3, 1],
        inner_sep="+",
    )
    assert result3 == "/a/d+b+c/e", result3

    key = "/left/leader/arm/pose"
    assert fragment_matching(
        key,
        frag_in={"leader", "arm"},
        frag_not={"right"},
        logic="and",
    )

    try:
        fragment_matching(
            key,
            frag_in={"leader", "arm"},
            frag_not={"arm"},
            logic="and",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for conflicting frag_in and frag_not")

    assert fragment_matching(
        key=key,
        frag_in={"leader", "arm"},
        frag_not={"right"},
        logic="or",
    )

    assert not fragment_matching(
        key=key,
        frag_in={"foo"},
        frag_not={"right"},
        logic="and",
    )

    assert fragment_matching(key=key, frag_not={"camera"})

    assert fragment_matching(key=key, startswith="/left")
    assert fragment_matching(key=key, endswith="/pose")

    result = call_multi_args(
        key,
        replace,
        [(("leader", "follower"), {}), (("arm", "leg"), {})],
    )
    assert result == "/left/follower/leg/pose", result
