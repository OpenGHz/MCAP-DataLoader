from pydantic import BaseModel, ConfigDict
from weakref import WeakValueDictionary
from mcap_data_loader.utils.array_like import Array
from mcap_data_loader.utils.basic import ForceSetAttr
from mcap_data_loader.utils.extra_itertools import recursive_map_reusable
from mcap_data_loader.pipelines.basis import Pipe, T
from typing import Optional, Dict, Set
from collections.abc import Mapping


class MeanStd(BaseModel, frozen=True):
    """Class to hold mean and standard deviation values."""

    model_config = ConfigDict(validate_assignment=True)

    mean: Optional[Array]
    """Mean value for standardization."""
    std: Optional[Array]
    """Standard deviation for standardization."""


class StandardizeConfig(BaseModel, frozen=True):
    """Configuration for Standardize pipeline."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = ""
    """Unique name of the pipeline. Used for caching instances."""
    statistics: Optional[Dict[str, MeanStd]] = None
    """Precomputed mean and std values corresponding to each key."""
    inverse: bool = False
    """Whether to perform inverse standardization."""
    include: Set[str] = set()
    """List of keys to include for standardization. If empty, all keys are included."""
    exclude: Set[str] = set()
    """List of keys to exclude from standardization."""
    depth: int = 0
    """Depth of recursion for nested iterable. 0 means no recursion.
    < 0 means recursion until a `Mapping` is encountered."""
    replace: bool = False
    """Whether to replace the original values with standardized values."""


class Standardize(Pipe[T]):
    _configs: WeakValueDictionary[str, StandardizeConfig] = WeakValueDictionary()

    def __init__(self, config: StandardizeConfig) -> None:
        self.config = config
        self._first_call = True

    def _on_first_call(self, iterable):
        config = self.config
        name = config.name
        if name in self._configs:
            self.get_logger().info("Reusing existing config for name: %s", name)
            update = {}
            for field in config.model_fields_set - {"name"}:
                update[field] = getattr(config, field)
            config = self._configs[name].model_copy(update=update)
            self.get_logger().info("Reused config fields: %s", update.keys())
        else:
            self._configs[name] = config
        self.config = config
        if config.statistics is None:
            self._process_none_stat(iterable)
        self._statistics = config.statistics
        self._replace = config.replace

    def _process_none_stat(self, iterable):
        method = "statistics"
        if hasattr(iterable, method):
            with ForceSetAttr(self.config) as cfg:
                cfg.statistics = getattr(iterable, method)()
        else:
            raise ValueError(
                "The `statistics` field must be provided in the config or the iterable must have a `statistics` method."
            )

    def transform(self, item: dict, inverse: bool) -> dict:
        """Transform a single data item by standardizing or inverse standardizing specified keys."""
        statistics = self._statistics
        new_item = item if self._replace else item.copy()
        for key in self._used_keys:
            stat = statistics[key]
            mean = stat.mean
            std = stat.std
            if inverse:
                new_item[key] = item[key] * std + mean
            else:
                new_item[key] = (item[key] - mean) / std
        return new_item

    def _transform(self, item: dict) -> dict:
        return self.transform(item, self.config.inverse)

    def on_call(self, iterable):
        if self._first_call:
            self._on_first_call(iterable)
            self._first_call = False
            config = self.config
            all_keys = self._statistics.keys()
            if config.include - all_keys:
                raise ValueError(
                    f"Included keys {config.include} are not present in statistics keys {all_keys}."
                )
            self._used_keys = (config.include or all_keys) - config.exclude
            self.get_logger().info("Keys to be standardized: %s", self._used_keys)
        return recursive_map_reusable(
            self._transform, iterable, self.config.depth, base_type=Mapping
        )


if __name__ == "__main__":
    import numpy as np
    import logging
    from pprint import pprint

    logging.basicConfig(level=logging.INFO)

    config = StandardizeConfig(
        statistics={
            "a": MeanStd(mean=np.array(5.0), std=np.array(2.0)),
            "b": MeanStd(mean=np.array([1.0, 2.0]), std=np.array([0.5, 0.5])),
            "c": MeanStd(mean=np.array(0.0), std=np.array(1.0)),
            "d": MeanStd(mean=np.array(10.0), std=np.array(2.0)),
        },
        include={"a", "b", "d"},
        exclude={"c", "d"},
    )
    pipeline = Standardize(config)
    data = [
        {
            "a": np.array(7.0),
            "b": np.array([2.0, 3.0]),
            "c": np.array(0.0),
            "d": np.array(12.0),
        },
        {
            "a": np.array(3.0),
            "b": np.array([0.0, 1.0]),
            "c": np.array(0.0),
            "d": np.array(8.0),
        },
    ]
    standardized = pipeline(data)
    print("---- Standardized Data ----")
    for item in standardized:
        pprint(item)
    # Inverse standardization
    inverse_config = StandardizeConfig(inverse=True)
    inverse_pipeline = Standardize(inverse_config)
    inversed = inverse_pipeline(standardized)
    print("---- Inversed Data ----")
    for item in inversed:
        pprint(item)
