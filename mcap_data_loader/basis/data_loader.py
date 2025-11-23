from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Optional, Union, Generic, TypeVar
from collections.abc import Iterable, Callable, Hashable, Mapping
from mcap_data_loader.utils.basic import StrEnum, NonIteratorIterable
from mcap_data_loader.pipelines import NamedPipelines
from mcap_data_loader.basis.cfgable import ConfigurableBasis
from enum import auto
from pprint import pformat
from toolz import pipe


T = TypeVar("T")


class Stage(StrEnum):
    TRAIN = auto()
    TEST = auto()
    INFER = auto()


class DataSourceConfig(BaseModel, frozen=True):
    """Configuration for a data source."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source: NonIteratorIterable
    """The data source (a dataset or live data stream)."""
    pipeline: Optional[Union[Hashable, Callable]] = None
    """The pipeline to use for this data source. If None, no pipeline will be applied."""


class DataLoaderConfig(BaseModel, frozen=True):
    """Configuration for data loader maker."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    sources: Dict[Union[Stage, str], Optional[DataSourceConfig]] = Field(min_length=1)
    """Configuration for data sources to be loaded. The key is the data loader name,
    which can be of type `Stage` or any string. If a `splitter` is provided
    in the `DataSourceConfig`, the item corresponding to the key will be populated
    and the sources will be updated by the returned dict from the `splitter`. Raise
    an error if there is a key conflict between the original keys and the keys returned 
    by the `splitter`."""


class DataLoaders(ConfigurableBasis, Generic[T]):
    """Base class for data loaders."""

    def __init__(self, config: DataLoaderConfig):
        self.config = config

        src_cfgs = config.sources
        data_loaders = {}
        for name, src_cfg in src_cfgs.items():
            if src_cfg is None:
                # skip if no data source config is provided,
                # mainly used to ignore a source in hydra yaml config
                continue
            result = self._apply_pipeline(src_cfg.source, src_cfg.pipeline)
            if isinstance(result, Mapping):
                data_loaders.update(result)
            else:
                data_loaders[name] = result
        self._data_loaders = data_loaders

    def on_configure(self):
        return True

    def _apply_pipeline(self, source, pipeline):
        if pipeline is None:
            return source
        elif callable(pipeline):
            return pipeline(source)
        else:
            return pipe(source, *NamedPipelines[pipeline])

    def __call__(self) -> Dict[Union[Stage, str], Iterable[T]]:
        """Get the data loaders."""
        return self._data_loaders

    def __getitem__(self, key: Union[Stage, str]) -> Iterable[T]:
        return self._data_loaders[key]

    def __repr__(self):
        return super().__repr__() + f"(\n{pformat(self._data_loaders)})"
