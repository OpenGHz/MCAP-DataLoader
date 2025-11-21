from mcap_data_loader.utils.basic import (
    NonIteratorIterable,
    is_not_implemented,
    not_implemented,
)
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.utils.extra_itertools import recursive_map, Reusablizer
from pydantic import BaseModel, NonNegativeInt, ConfigDict, validate_call
from typing import TypeVar, Dict, Union, Optional, final
from collections.abc import Iterator, Callable, Iterable
from toolz import compose_left, pipe
from functools import partial
from pprint import pformat


T = TypeVar("T")


class Pipe(CallerBasis[T]):
    """Base class for all pipes. A pipe can be viewed
    as a special type of caller that can only accept one iterable
    parameter and return an iterable value and can only be called once.
    """

    recall = None
    """If True, a new instance of the pipe will be created
    whenever the pipe is called. If False, the same instance
    will be reused. If None, the behavior will be determined
    based on whether the pipe implements `__iter__` method: 
    if `__iter__` is implemented, recall will be set to False,
    otherwise it will be set to True."""

    @final
    def config_post_init(self):
        super().config_post_init()
        self._called = False
        if self.__class__.recall is None:
            if is_not_implemented(self.__iter__):
                self.__class__.recall = False
            else:
                self.__class__.recall = True

    @final
    @validate_call(validate_return=True)
    def __call__(self, iterable: NonIteratorIterable) -> NonIteratorIterable[T]:
        """Initialize the pipe with the given iterable."""
        # TODO: should remove the final checker to allow the subclass to
        # override for better typing?
        if self.__class__.recall is ...:
            if self._called:
                raise RuntimeError("This pipe can only be called once.")
            self._called = True
        elif self.__class__.recall:
            self = self.copy()
        return self.on_call(iterable)

    def on_call(self, iterable: NonIteratorIterable) -> NonIteratorIterable[T]:
        """Hook called after the pipe is called."""
        self._iterable = iterable
        return self

    @not_implemented
    def __iter__(self) -> Iterator[T]:
        """Yield items from the pipe."""


class PipelineConfig(BaseModel, frozen=True):
    """Configuration for a pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipes: Union[Dict[float, Union[Dict[float, Optional[Callable]], NonNegativeInt]]]
    """Multiple grouped pipes to be connected with diverter."""


class Pipeline(CallerBasis[Iterable[T]]):
    """Pipeline connects multiple pipes together to work together to complete tasks.
    In a pipeline, the output of one command is passed to the input of the next command,
    so that a series of commands can work together to complete more complex tasks."""

    def __init__(self, config: PipelineConfig) -> None:
        self._pipes = config.pipes
        self._chained = None

    def __call__(self, iterable: NonIteratorIterable):
        if self._chained is None:
            depth = None
            chained = []
            for group_i in sorted(self._pipes.keys()):
                value = self._pipes[group_i]
                if isinstance(value, int):
                    if depth is not None:
                        raise ValueError("Duplicate depth specification.")
                    depth = value
                else:
                    a_pipeline = []
                    for pipe_i in sorted(value.keys()):
                        a_pipe = value[pipe_i]
                        if a_pipe is not None:
                            a_pipeline.append(a_pipe)
                    composed = compose_left(*a_pipeline)
                    if depth is None:
                        chained.append(composed)
                    else:
                        if depth == 0:
                            func = partial(map, composed)
                        else:
                            func = partial(recursive_map, composed, depth=depth)
                        chained.append(Reusablizer(func))
                    depth = None
            self._chained = chained
            self.get_logger().info(f"Chained pipeline:\n{pformat(self._chained)}")
        return pipe(iterable, *chained)


PipelineType = Callable[[Iterable], Iterable]
"""Type alias for pipeline functions. Returning an Iterator 
is allowed because some custom classes, although instances 
of iterator type, can reset their internal state at the end 
of the iteration to achieve an effect similar to a regular 
iterable, such as torchdata nodes."""
