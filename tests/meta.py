from pydantic import BaseModel
from abc import ABCMeta
from typing import get_type_hints, Any, TypeVar

T = TypeVar("T")


class Config(BaseModel):
    param: int = 0


class PreprocessMeta(type):
    def __call__(cls, config, **kwargs):
        print(cls.__annotations__)
        config_type = get_type_hints(cls.__init__).get("config", None)
        print(f"{config_type=}")
        instance = super().__call__(config)
        instance.config = config
        return instance

    def show_info(cls):
        print(f"Class {cls.__name__} with PreprocessMeta")


class Base(metaclass=PreprocessMeta):
    # config: Config

    def __init__(self, config: T):
        """Initialize with a config, if needed."""
        self.config = config


class BaseABCMeta(ABCMeta, PreprocessMeta):
    pass


class OtherConfig(BaseModel):
    param: int = 0


class MyClassWithInit(Base):
    def __init__(self, config: OtherConfig):
        self.config = config

    def configure(self):
        self.config


class MyClassWithoutInit(Base):
    config: Config

    def print_config(self):
        print(f"Config: {self.config}")


obj = MyClassWithInit(Config(param=10), param=1)
print(obj.config)

obj2 = MyClassWithoutInit(Config(param=20))
obj2.print_config()
