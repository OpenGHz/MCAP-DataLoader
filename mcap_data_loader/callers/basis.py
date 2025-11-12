from abc import abstractmethod
from typing import Tuple, Literal, Generic, TypeVar, List
from collections.abc import Callable
from pydantic import BaseModel, Field
from mcap_data_loader.basis.cfgable import ConfigurableBasis, ConfigT


ReturnT = TypeVar("T")
CallT = TypeVar("CallT", bound=Callable)


class CallerBasis(ConfigurableBasis[ConfigT], Generic[ConfigT, ReturnT]):
    def reset(self) -> None:
        """Reset the internal state of the caller, if any."""

    def on_configure(self):
        return True

    @abstractmethod
    def __call__(self, *args, **kwds) -> ReturnT:
        """Call the caller with the given inputs."""


class HorizonConfig(BaseModel):
    input: int = 1
    output: int = 1


class MockCallerConfig(BaseModel):
    output_type: Literal["ndarray", "Tensor", "list"] = "list"
    horizon: HorizonConfig = HorizonConfig()


class MockCaller(CallerBasis[MockCallerConfig, ReturnT]):
    """A mock caller that outputs a sequence of numbers based on the input."""

    def on_configure(self):
        if self.config.output_type == "ndarray":
            import numpy as xp
        elif self.config.output_type == "Tensor":
            import torch as xp
        self._xp = xp
        return True

    def reset(self):
        """Do nothing"""

    def __call__(self, data: Tuple[float]):
        lis = [[[data[0] + i] for i in range(self.config.horizon.output)]]
        if self.config.output_type == "list":
            return lis
        elif self.config.output_type == "ndarray":
            return self._xp.array(lis)
        elif self.config.output_type == "Tensor":
            return self._xp.tensor(lis)


class CallerEnsembleConfig(BaseModel, Generic[CallT]):
    callables: List[CallT] = Field(min_length=1)
    """Tuple of callers to be called in ensemble."""


CET = TypeVar("CEConfig", bound=CallerEnsembleConfig)


class CallerEnsembleBasis(CallerBasis[CET, ReturnT]):
    """A caller that ensembles multiple callers' outputs."""

    def on_configure(self):
        for func in self.config.callables:
            if isinstance(func, CallerBasis):
                if not func.configure():
                    self.get_logger().error(f"Failed to configure callable: {func}")
                    return False
        return True

    def reset(self):
        """Reset the internal state of the caller, if any."""
        for func in self.config.callables:
            if isinstance(func, CallerBasis):
                func.reset()
