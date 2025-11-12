from pydantic import BaseModel, PositiveInt
from collections.abc import Callable
from typing import Optional, Literal, TypeVar, MutableSequence
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from mcap_data_loader.callers.basis import (
    CallerEnsembleConfig,
    CallerEnsembleBasis,
)
from mcap_data_loader.utils.array_like import (
    ArrayInfo,
    array_namespace,
    get_namespace_by_name,
    get_array_type_by_ns_name,
    get_tensor_device_auto,
)


CallableT = TypeVar("CallableT", bound=Callable)
T = TypeVar("T")


class ScalarsToContainerConfig(BaseModel):
    """Configuration for converting scalar outputs to a container."""

    xp: Literal["list", "numpy", "torch"] = "list"
    """The container type to convert scalars into."""
    dtype: str = ""
    """The data type of the container elements."""
    device: str = ""
    """The device of the container."""

    def model_post_init(self, context):
        if self.xp != "list":
            if not self.dtype:
                if self.xp == "numpy":
                    self.dtype = "float64"
                elif self.xp == "torch":
                    self.dtype = "float32"
            if self.xp == "numpy":
                self.device = "cpu"
            elif self.xp == "torch" and not self.device:
                self.device = get_tensor_device_auto()


class MultiCallerConfig(CallerEnsembleConfig):
    """Configuration for MultiCaller"""

    num_workers: Optional[PositiveInt] = 1
    """Number of worker threads to use. 1 means no parallelism (no executor).
    None means using as many workers as callables."""
    mode: Literal["thread", "process"] = "thread"
    """Mode of parallelism: thread or process."""
    scalar_container: ScalarsToContainerConfig = ScalarsToContainerConfig()
    """The container type for scalar outputs (int or float). Otherwise, the output
    type will be used if it is array-like. If the output is not array-like, a list
    container will be used."""


class MultiCaller(CallerEnsembleBasis[MutableSequence[T]]):
    """A caller that calls multiple callables in sequence and aggregates their outputs."""

    def __init__(self, config: MultiCallerConfig):
        self.config = config
        self._callables = config.callables
        max_workers = config.num_workers or len(config.callables)
        if max_workers > 1:
            self._executor = (
                ThreadPoolExecutor(max_workers)
                if config.mode == "thread"
                else ProcessPoolExecutor(max_workers)
            )
            self._call = self._call_in_parallel
        else:
            self._call = self._call_in_sequence
        self._first_call = True

    def _first_setup(self, *args, **kwds):
        info_set = set()
        has_scalar = False
        for func in self._callables:
            output = func(*args, **kwds)
            if isinstance(output, (int, float)):
                has_scalar = True
            else:
                info_set.add(ArrayInfo.from_array(output))
        if len(info_set) > 1:
            print(f"The outputs of the callables have different array info: {info_set}")
        num_calls = len(self._callables)
        scalar_cfg = self.config.scalar_container
        if len(info_set) > 1 or (has_scalar and scalar_cfg.xp == "list"):
            self._p_outputs = [None] * num_calls
        else:
            if has_scalar:
                info = ArrayInfo(
                    arr_type=get_array_type_by_ns_name(scalar_cfg.xp),
                    shape=(),
                    dtype=scalar_cfg.dtype,
                    device=scalar_cfg.device,
                )
                xp = get_namespace_by_name(scalar_cfg.xp)
            else:
                info: ArrayInfo = info_set.pop()
                xp = array_namespace(output)
            self._p_outputs = xp.zeros(
                (num_calls,) + info.shape,
                dtype=getattr(xp, info.dtype),
                device=info.device,
            )

    def _call_in_sequence(self, *args, **kwds):
        for i, func in enumerate(self._callables):
            output = func(*args, **kwds)
            self._p_outputs[i] = output
        return self._p_outputs

    def _call_in_parallel(self, *args, **kwds):
        future_to_index = {
            self._executor.submit(func, *args, **kwds): idx
            for idx, func in enumerate(self._callables)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            self._p_outputs[idx] = future.result()
        return self._p_outputs

    def __call__(self, *args, **kwds):
        """Call all the callables in sequence and aggregate their outputs."""
        if self._first_call:
            self._first_setup(*args, **kwds)
            self._first_call = False
        return self._call(*args, **kwds)


if __name__ == "__main__":
    callables = [
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x**2,
    ]
    # import numpy as np
    # import torch

    # xp = torch  # np or torch
    # device = "cpu" if xp is np else torch.device("cuda")

    # callables = [
    #     lambda x: xp.ones((2, 2), device=device) * (x + 1),
    #     lambda x: xp.ones((2, 2), device=device) * x * 2,
    #     lambda x: xp.ones((2, 2), device=device) * x**2,
    # ]

    config = MultiCallerConfig(
        callables=callables,
        num_workers=None,
        mode="thread",
        scalar_container=ScalarsToContainerConfig(xp="torch"),
    )
    multi_caller = MultiCaller(config=config)
    multi_caller.on_configure()
    result = multi_caller(3)
    print(result)  # Expected output: [4, 6, 9]
    print(ArrayInfo.from_array(result))
    # Test with no parallelism
    config_no_parallel = MultiCallerConfig(callables=callables, num_workers=1)
    multi_caller_no_parallel = MultiCaller(config=config_no_parallel)
    multi_caller_no_parallel.on_configure()
    result_no_parallel = multi_caller_no_parallel(3)
    print(result_no_parallel)  # Expected output: [4, 6, 9]
