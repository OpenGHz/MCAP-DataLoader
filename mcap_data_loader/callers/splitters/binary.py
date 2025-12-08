from mcap_data_loader.callers.basis import SplitterBasis
from mcap_data_loader.utils.extra_itertools import take_skip
from pydantic import BaseModel, NonNegativeFloat, PositiveInt
from typing import Tuple, Optional
from random import Random


class BinarySplitterConfig(BaseModel, frozen=True):
    """Configuration for BinarySplitter. Either `ratio`, `take_skip` or `grouped_ratio`
    should be provided. The last two methods are suitable for repeatedly collecting data
    N times under the same settings, then changing to the next setting and continuing to
    collect data. After traversing M settings, there are a total of N * M episodes. During
    training, this partitioning method ensures that the validation set covers all different
    settings."""

    keys: Tuple[str, str] = ("first", "second")
    """Tuple containing the names of the two keys to which the data will be split."""
    ratio: Optional[NonNegativeFloat] = None
    """Ratio of the first key data to the total data."""
    take_skip: Tuple[PositiveInt, PositiveInt] = ()
    """Tuple containing the number of items to take and skip `(first_num, second_num)`. 
    This means that `first_num` data points are taken from the `episode` list, followed 
    by `second_num` data points, and so on. The final ratio of the training set to the 
    validation set is approximately `first_num:second_num`"""
    grouped_ratio: Tuple[PositiveInt, NonNegativeFloat] = ()
    """Tuple containing the number of items in a group and the ratio of the first key data 
    in each group `(group_num, ratio)`. This means that the input data is  divided into groups 
    of `group_num` data points, and in each group, the first `ratio` portion of data points are 
    assigned to the first key, while the remaining data points are assigned to the second key."""
    in_order: bool = True
    """Whether to take data in order when using `take_skip` or `grouped_ratio` method. It may be
    slower to load data in order."""
    seed: Optional[int] = None
    """Random seed for shuffling data. If None, no shuffling is applied.
    This takes precedence over the `in_order` setting."""

    def model_post_init(self, context):
        result = sum(
            [
                self.ratio is not None,
                bool(self.take_skip),
                bool(self.grouped_ratio),
            ]
        )
        if result != 1:
            raise ValueError(
                "Exactly one of `ratio`, `take_skip`, or `grouped_ratio` must be provided."
                f" Got {self}."
            )


class BinarySplitter(SplitterBasis):
    """A caller that splits the input data into two parts based on the given configuration."""

    def __init__(self, config: BinarySplitterConfig):
        if config.ratio is not None:
            self._ratio = config.ratio
            split_method = self._split_by_ratio
        else:
            if config.take_skip:
                first_num, second_num = config.take_skip
            else:
                group_size, ratio = config.grouped_ratio
                first_num = int(group_size * ratio)
                second_num = group_size - first_num
            self._first_num = first_num
            self._second_num = second_num
            split_method = self._split_by_take_skip
        self._split_method = split_method
        self._keys = config.keys
        self._in_order = config.in_order
        self._random = None if config.seed is None else Random(config.seed)

    def _split_by_ratio(self, data_list: list):
        total_len = len(data_list)
        first_num = int(total_len * self._ratio)
        return data_list[:first_num], data_list[first_num:]

    def _split_by_take_skip(self, data_list: list):
        # TODO: Testing whether `in_order=True` significantly slows down loading speed
        return take_skip(
            data_list, self._first_num, self._second_num, in_order=self._in_order
        )

    def __call__(self, data):
        data_list = list(data)
        total_len = len(data_list)
        if total_len <= 1:
            self.get_logger().warning(
                f"Data length {total_len} is too small to split by ratio."
                f" Returning original data as the first part and empty list as the second part."
            )
            return data_list, []
        # self.get_logger().info(f"Splitting data {data_list}.")
        if self._random is not None:
            self._random.shuffle(data_list)
        return dict(zip(self._keys, self._split_method(data_list)))


if __name__ == "__main__":
    data = range(10)
    Config = BinarySplitterConfig
    for config in [
        Config(ratio=0.3),
        Config(take_skip=(2, 3)),
        Config(grouped_ratio=(5, 0.4)),
    ]:
        splitter = BinarySplitter(config)
        result = splitter(data)
        print(result)
