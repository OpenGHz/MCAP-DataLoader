from mcap_data_loader.utils.basic import NonIteratorIterable
from pydantic import validate_call
from typing import List, TypeVar, Generic, final
from typing_extensions import Self
from collections.abc import Generator
from abc import ABC, abstractmethod


T = TypeVar("T")


class Pipeline(ABC, Generic[T]):
    """Base class for all pipelines."""

    @final
    @validate_call
    def wrap(self, iterables: List[NonIteratorIterable]) -> Self:
        self._iterables = iterables
        return self

    @abstractmethod
    def __iter__(self) -> Generator[T]:
        """Yield items from the pipeline."""
