from typing import Generic, TypeVar
from abc import ABC
from mcap_data_loader.utils.basic import resolve_generic_type
from dataclasses import dataclass
import inspect


@dataclass
class PConfig:
    param1: int


@dataclass
class CConfig(PConfig):
    param2: str


T = TypeVar("T")
print(T.__bound__)
TBound = TypeVar("TBound", bound=PConfig)
print(f"{TBound.__bound__=}")


class PBase(Generic[TBound]):
    def __init__(self, config: TBound) -> None:
        self.config = config


class CBase(PBase[CConfig]):
    def print_config(self):
        print(f"param1: {self.config.param1}, param2: {self.config.param2}")


class Basis(Generic[T]):
    config: T

    def __init__(self, config: T) -> None:
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        generic_type = resolve_generic_type(cls, Basis)
        generic_type = (
            generic_type.__bound__
            if isinstance(generic_type, TypeVar)
            else generic_type
        )
        if generic_type:
            print(f"Registered generic type for {cls.__name__}: {generic_type}")
        cls._generic_type = generic_type


StringBound = TypeVar("StringBound", bound=str)


class StringPair(ABC, Basis[StringBound]):
    pass


class CustomBasis(ABC, Basis[T]):
    pass


class AnyCustom(CustomBasis):
    def __init__(self, config: int):
        super().__init__(config)
        self.config = config

    def print(self):
        print(self.config)


print(AnyCustom._generic_type)  # T
print(isinstance(AnyCustom._generic_type, TypeVar))  # True


class CustomDict(CustomBasis[dict]):
    pass


class CustomDictSame(CustomDict):
    pass


class CustomDictChild(CustomDict, Basis[float]):
    def print(self):
        print(self._generic_type)
        print(f"Config: {self.config}")


class CustomDictChild0(CustomDict, Basis[float]):
    def __init__(self, config: float):
        super().__init__(config)
        self.config = config

    def print(self):
        print(self._generic_type)
        print(f"Config: {self.config}")


# class CustomDictChild(CustomDict):
#     config: float

#     def print(self):
#         print(self._generic_type)
#         print(f"Config: {self.config}")


child = CustomDictChild(config=3.14)
child.print()


class CustomDictChildFisrt(Basis[float], CustomDict):
    def print(self):
        print(f"Config: {self.config}")


class CustomDictChildDeep(CustomDictChild, Basis[int]):
    pass


class SomeCustom(AnyCustom[int]):
    pass
