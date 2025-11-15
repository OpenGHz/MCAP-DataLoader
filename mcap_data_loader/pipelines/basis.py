from mcap_data_loader.utils.basic import NonIteratorIterable
from mcap_data_loader.callers.basis import CallerBasis
from pydantic import validate_call
from typing import Any, TypeVar, final
from typing_extensions import Self
from collections.abc import Iterator
from abc import abstractmethod


T = TypeVar("T")


class Pipeline(CallerBasis[T]):
    """Base class for all pipelines. A pipeline can be viewed
    as a special type of caller that can only accept one iterable
    parameter and return an iterable value and can only be called once.
    """

    @final
    def config_post_init(self):
        super().config_post_init()
        self._called = False

    @final
    @validate_call
    def __call__(self, iterable: NonIteratorIterable[Any]) -> Self:
        """Initialize the pipeline with the given iterable."""
        if self._called:
            raise RuntimeError("A pipeline instance can only be called once.")
        self._called = True
        self._iterable = iterable
        return self

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Yield items from the pipeline."""
