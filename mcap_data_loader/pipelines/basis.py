from mcap_data_loader.utils.basic import (
    NonIteratorIterable,
    is_not_implemented,
    not_implemented,
)
from mcap_data_loader.callers.basis import CallerBasis
from pydantic import BaseModel, validate_call
from toolz import pipe
from typing import TypeVar, Dict, Union, Optional, Hashable, List, final
from collections.abc import Iterator, Callable, Iterable
from pprint import pformat
from logging import getLogger


T = TypeVar("T")


class Pipe(CallerBasis[T]):
    """Base class for all pipes. A pipe can be viewed
    as a special type of caller that can only accept one iterable
    parameter and return an iterable value and can only be called once.
    """

    recall: Optional[bool] = None
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


PipelineType = Dict[float, Optional[Union[Callable, Hashable]]]
NamedPipelineValue = Optional[Dict[float, Optional[Callable]]]
NamedPipelineType = Dict[Hashable, NamedPipelineValue]
NamedPipelines: Dict[Hashable, List[Callable]] = {}
"""Pipelines with names which will be shared across `Pipeline` instances. 
Often, a pipeline may contain nested  sub-pipelines, and we want this 
nesting to be easily removed. This enables flexible reuse, including 
deleting existing nested pipelines and updating them to be non-nested."""


def _chain_pipes(pipeline: PipelineType) -> Callable:
    """Compose multiple pipes into a single callable."""
    chained = []
    for pipe_key in sorted(pipeline.keys()):
        a_pipe = pipeline[pipe_key]
        if callable(a_pipe):
            chained.append(a_pipe)
        elif a_pipe is not None:
            pipe_chain = NamedPipelines[a_pipe]
            if pipe_chain is not None:
                chained.extend(pipe_chain)
    return chained


@validate_call
def register_named_pipelines(
    named_pipelines: NamedPipelineType = {}, **kwargs: NamedPipelineValue
) -> None:
    """Register named pipelines into the given named_pipelines dict."""
    keys = named_pipelines.keys() | kwargs.keys()
    logger = getLogger("register_named_pipelines")
    logger.info(f"Registering: {list(keys)}")
    if keys & NamedPipelines.keys():
        logger.warning(
            f"Key conflict between existing: "
            f"{list(NamedPipelines.keys())} and new: "
            f"{list(keys)}. Overwriting existing pipelines."
        )
    elif None in keys:
        raise ValueError("Pipeline name cannot be None.")
    for name, dict_pipeline in (named_pipelines | kwargs).items():
        NamedPipelines[name] = (
            _chain_pipes(dict_pipeline) if dict_pipeline is not None else None
        )


class PipelineConfig(BaseModel, frozen=True):
    """Configuration for a pipeline."""

    # model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: PipelineType
    """Multiple pipes to be connected. The keys are used to
    determine the order of the pipes. The values can be either
    the pipe callable or the name of the pipeline in `named_pipelines`."""


class Pipeline(CallerBasis[Iterable[T]]):
    """Pipeline connects multiple pipes together to work together to complete tasks.
    In a pipeline, the output of one command is passed to the input of the next command,
    so that a series of commands can work together to complete more complex tasks."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._chained = _chain_pipes(config.pipeline)
        self.get_logger().info(f"Chained pipeline:\n{pformat(self._chained)}")

    def __call__(self, iterable: NonIteratorIterable):
        """Apply the pipeline to the given iterable."""
        return pipe(iterable, *self._chained)


PipeType = Callable[[Iterable], Iterable]
"""Type alias for pipeline functions. Returning an Iterator 
is allowed because some custom classes, although instances 
of iterator type, can reset their internal state at the end 
of the iteration to achieve an effect similar to a regular 
iterable, such as torchdata nodes."""
