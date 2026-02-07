from enum import auto
from typing import TypeVar, Generic, Dict, Tuple, Optional, List, Union
from collections.abc import Hashable, Callable, Iterable
from typing_extensions import TypedDict
from copy import deepcopy, copy
from statistics import mean
from mcap_data_loader.basis.data_types import StrEnum


T = TypeVar("T")
ReturnT = TypeVar("ReturnT")
KeyT = TypeVar("KeyT", bound=Hashable)
DataT = TypeVar("DataT")


def sum_auto_start(iterable: Iterable[T]) -> T:
    """Sum the items in the iterable, starting from the first item."""
    iterator = iter(iterable)
    total = copy(next(iterator))
    for item in iterator:
        total += item
    return total


class OutputMode(StrEnum):
    """Mode to specify how to store the results of the map_dict function."""

    INPLACE = auto()
    """Modify the input dictionary in place."""
    COPY = auto()
    """Modify a copy of the input dictionary."""
    VALUE_COPY = auto()
    """Modify a copy of the input dictionary with each value copied."""
    DEEP_COPY = auto()
    """Modify a deep copy of the input dictionary."""
    EMPTY = auto()
    """Use an empty dictionary to store the results so that the keys not processed will be dropped."""

    @staticmethod
    def inplace(data):
        return data

    @staticmethod
    def copy(data):
        return copy(data)

    @staticmethod
    def value_copy(data):
        return copy_dict_data_stamped(data)

    @staticmethod
    def deep_copy(data):
        return deepcopy(data)

    @staticmethod
    def empty(data):
        return {}


OUTPUT_MODE_FUNC_MAP = {
    OutputMode.INPLACE: OutputMode.inplace,
    OutputMode.COPY: OutputMode.copy,
    OutputMode.VALUE_COPY: OutputMode.value_copy,
    OutputMode.DEEP_COPY: OutputMode.deep_copy,
    OutputMode.EMPTY: OutputMode.empty,
}


class DataStamped(TypedDict, Generic[T]):
    t: int
    """Timestamp of the data."""
    data: T
    """The actual data."""

    @staticmethod
    def map_dict(
        func: Callable[[DataT], ReturnT],
        data: Dict[KeyT, "DataStamped[DataT]"],
        keys: Optional[Iterable[KeyT]] = None,
        output: Union[dict, OutputMode] = OutputMode.INPLACE,
    ) -> Dict[KeyT, "DataStamped[ReturnT]"]:
        """map a function to the data part of each DataStamped in the dictionary.
        Args:
            func (Callable[[DataT], ReturnT]): The function to apply to the data part.
            data (Dict[KeyT, DataStamped[DataT]]): The input dictionary.
            keys (Optional[Iterable[KeyT]], optional): The keys to process. If None, process all keys. Defaults to None.
            output (Union[dict, OutputMode], optional): The output dictionary to store the results or the mode to use. Defaults to OutputMode.INPLACE.
        Returns:
            Dict[KeyT, DataStamped[ReturnT]]: The output dictionary with the processed data.
        """
        result = (
            OUTPUT_MODE_FUNC_MAP[output](data)
            if isinstance(output, OutputMode)
            else output
        )
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
        """merge multiple DataStamped into one.
        Args:
            values (Iterable[DataStamped[DataT]]): The input DataStamped objects.
            d_method (Callable[[List[DataT]], ReturnT], optional): The method to merge the data part. Defaults to sum_auto_start.
            t_method (Callable[[List[int]], int], optional): The method to merge the time part. Defaults to mean.
        Returns:
            DataStamped[ReturnT]: The merged DataStamped object.
        """
        time_list = []
        data_list = []
        for item in values:
            time_list.append(item["t"])
            data_list.append(item["data"])
        return {"t": int(t_method(time_list)), "data": d_method(data_list)}

    @staticmethod
    def concatenate(
        values: Iterable["DataStamped[Iterable[DataT]]"],
    ) -> Tuple[List[int], List[DataT]]:
        """Concatenate multiple DataStamped with list data into one.
        Args:
            values (Iterable[DataStamped[Iterable[DataT]]]): The input DataStamped objects.
        Returns:
            Tuple[List[int], List[DataT]]: The concatenated time list and data list.
        """
        time_list = []
        data_list = []
        for item in values:
            time_list.append(item["t"])
            data_list.extend(item["data"])
        return time_list, data_list

    @staticmethod
    def create(data: T, t: int = 0) -> "DataStamped[T]":
        return {"t": t, "data": data}


DictDataStamped = Dict[str, DataStamped[T]]


def copy_dict_data_stamped(
    data: DictDataStamped[T], deep: bool = False
) -> DictDataStamped[T]:
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


if __name__ == "__main__":
    import numpy as np
    import time

    data = {
        "a": {"t": 1, "data": [1, 2]},
        "b": {"t": 2, "data": [3, 4]},
    }
    start = time.perf_counter()
    result = DataStamped.map_dict(np.array, data)
    print("Time taken:", time.perf_counter() - start)
    for value in result.values():
        print(value["t"])
        print(value["data"].shape)
