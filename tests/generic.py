from typing import Generic, TypeVar
from abc import ABC
from mcap_data_loader.utils.basic import resolve_generic_type
from dataclasses import dataclass


@dataclass
class PConfig:
    param1: int


@dataclass
class CConfig(PConfig):
    param2: str


T = TypeVar("T")
TBound = TypeVar("TBound", bound=PConfig)


class PBase(Generic[TBound]):
    def __init__(self, config: TBound) -> None:
        self.config = config


class CBase(PBase[CConfig]):
    def print_config(self):
        print(f"param1: {self.config.param1}, param2: {self.config.param2}")


class Basis(Generic[T]):
    def __init__(self, first: T) -> None:
        self.first = first

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._generic_type = resolve_generic_type(cls, Basis)
        if cls._generic_type and not isinstance(cls._generic_type, TypeVar):
            print(f"Registered generic type for {cls.__name__}: {cls._generic_type}")


class StringPair(ABC, Basis[str]):
    pass


class CustomBasis(ABC, Basis[T]):
    pass


class AnyCustom(CustomBasis):
    pass


print(AnyCustom._generic_type)  # T
print(isinstance(AnyCustom._generic_type, TypeVar))  # True


class CustomDict(CustomBasis[dict]):
    pass


class CustomDictChild(CustomDict, Basis[float]):
    pass


class SomeCustom(AnyCustom[int]):
    pass
