from pydantic import BaseModel, ConfigDict, ImportString
from typing import Union, Generic, Dict, Any
from collections.abc import Callable
from mcap_data_loader.callers.basis import CallerBasis, ReturnT
from toolz import curry


class CurryConfig(BaseModel, Generic[ReturnT], frozen=True):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )

    callable: Union[ImportString[Callable[..., ReturnT]], Callable[..., ReturnT]]
    args: tuple = ()
    kwargs: Dict[str, Any] = {}


class Curry(CallerBasis[ReturnT]):
    def __init__(self, config: CurryConfig[ReturnT]):
        if not config.args and not config.kwargs:
            self._curried = config.callable
        else:
            self._curried = curry(config.callable, *config.args, **config.kwargs)

    def __call__(self, *args, **kwds):
        return self._curried(*args, **kwds)


if __name__ == "__main__":
    caller = Curry(
        CurryConfig(
            # callable=lambda x, y=1: x + y,
            callable="operator.add",
            args=(2,),
            kwargs={"y": 3},
        )
    )
    assert caller() == 5
    try:
        caller(4)
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
