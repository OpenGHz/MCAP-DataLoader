import inspect
from abc import ABC, abstractmethod
from dataclasses import asdict, replace
from logging import getLogger
from typing import Union, Optional, Type, final
from pydantic import BaseModel


ConfigType = Optional[Union[BaseModel, Type[BaseModel]]]


class InitConfigMixin:
    def __init__(self, config: ConfigType = None, **kwargs) -> None:
        """Base class for configurable components.
        Args:
            config: Configuration object, typically a pydantic BaseModel or a dataclass.
            **kwargs: Additional keyword arguments to override config fields.
        """
        # mainly used by yaml config, e.g. hydra
        if config is None or isinstance(config, type):
            config_type = config or self.__annotations__.get("config", None)
            if not config_type:
                raise ValueError(
                    "`config` must be annotated at the top level class if not provided as an arg."
                )
            # self.get_logger().info(f"{kwargs}")
            config = config_type(**kwargs)
            # check pydantic extra kwargs
            if isinstance(config, BaseModel):
                extra = kwargs.keys() - config.__class__.model_fields.keys()
                if extra:
                    self.get_logger().warning(
                        f"Extra fields {extra} found in config, which will be ignored."
                    )
        else:  # mainly used by instancing manually
            if kwargs:  # rarely used
                if isinstance(config, BaseModel):
                    config = config.model_copy(update=kwargs)
                    # re-validate
                    config = config.model_validate(config.model_dump(warnings="none"))
                else:  # dataclass
                    config = replace(config, **kwargs)
        self.config = config
        self._configured = False
        self.on_init()

    def on_init(self) -> None:
        """Callback to be called when initializing"""

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)


class ConfigurableBasis(ABC, InitConfigMixin):
    @final
    def configure(self) -> bool:
        if self._configured:
            raise RuntimeError("Already configured")
        class_type = self.__annotations__.get("interface", None)
        if class_type is not None:
            self._create_interface(class_type)
        else:
            self.interface = None
        self._configured = self.on_configure()
        return self._configured

    @abstractmethod
    def on_configure(self) -> bool:
        """Callback to be called when configuring"""
        raise NotImplementedError

    @final
    @property
    def configured(self) -> bool:
        return self._configured

    def _create_interface(self, class_type: Type):
        """Create the interface instance based on the config and the class type annotation.
        The subclasses can override this method if needed.
        """
        sig = inspect.signature(class_type)
        if "config" in sig.parameters.keys():
            self.interface = class_type(config=self.config)
        else:
            # convert the first level config to dict
            if isinstance(self.config, BaseModel):
                # dict(self.config) has some bugs
                # so we use the following way
                cfg_dict = {
                    k: getattr(self.config, k)
                    for k in self.config.__class__.model_fields.keys()
                }
            else:  # dataclass
                # TODO: error when using nested dataclass
                cfg_dict = asdict(self.config)
            com_keys = cfg_dict.keys() & sig.parameters.keys()
            self.interface = class_type(**{key: cfg_dict[key] for key in com_keys})


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    class MyConfig(BaseModel):
        param1: int = 10
        param2: str = "default"

    class MyComponent(ConfigurableBasis):
        config: MyConfig

        def on_configure(self) -> bool:
            self.get_logger().info(f"Configuring with {self.config}")
            return True

    comp = MyComponent(param1=20)
    comp.configure()
