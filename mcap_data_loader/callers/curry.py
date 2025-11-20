from pydantic import BaseModel, ConfigDict
from typing import Union, Generic, Dict, Any
from collections.abc import Callable
from mcap_data_loader.utils.basic import get_class_type, force_set_attr
from mcap_data_loader.callers.basis import CallerBasis, ReturnT
from toolz import curry


class CurryConfig(BaseModel, Generic[ReturnT], frozen=True):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )

    callable: Union[str, Callable[..., ReturnT]]
    args: tuple = ()
    kwargs: Dict[str, Any] = {}

    @force_set_attr
    def model_post_init(self, context):
        if isinstance(self.callable, str):
            self.callable = get_class_type(self.callable)


class Curry(CallerBasis[ReturnT]):
    def __init__(self, config: CurryConfig[ReturnT]):
        self._curried = curry(config.callable, *config.args, **config.kwargs)

    def __call__(self, *args, **kwds):
        return self._curried(*args, **kwds)


if __name__ == "__main__":
    caller = Curry(
        CurryConfig(
            callable=lambda x, y=1: x + y,
            args=(2,),
            kwargs={"y": 3},
        )
    )
    assert caller() == 5
    try:
        caller(4)
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
