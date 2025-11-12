from abc import ABC, abstractmethod
from dataclasses import asdict, replace, is_dataclass
from logging import getLogger
from typing import Union, Optional, Type, Dict, Any, Literal, Generic, TypeVar, final
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from pathlib import Path
from mcap_data_loader.utils.basic import (
    get_class_type,
    resolve_generic_type,
    DataClassProto,
)
import inspect
import yaml
import json
import pickle


OtherCfgType = Optional[Union[Type[Union[DataClassProto, BaseModel]], Dict[str, Any]]]
ConfigT = TypeVar("ConfigT", bound=Union[BaseModel, DataClassProto])


class InitConfigMixin(Generic[ConfigT]):
    def __init__(self, config: Union[ConfigT, OtherCfgType] = None, **kwargs) -> None:
        """Base class for configurable components.
        Args:
            config: Configuration object, typically a pydantic BaseModel or a dataclass.
            **kwargs: Additional keyword arguments to override config fields.
        """
        # mainly used by yaml config, e.g. hydra
        if config is None or isinstance(config, type):
            config_type = config or self.resolve_config_type()
            if (not config_type) or isinstance(config_type, TypeVar):
                cls_path = kwargs.pop("_target_", None)
                if cls_path is not None:
                    config_type = get_class_type(cls_path)
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
            if isinstance(config, dict):
                config_type = self.resolve_config_type()
                if not config_type:
                    cls_path = config.pop("_target_", None)
                    if cls_path is not None:
                        config_type = get_class_type(cls_path)
                    raise ValueError(
                        "`config` must be annotated at the top level class if not provided as an arg."
                    )
                config = config_type(**config)
            if kwargs:  # rarely used
                if isinstance(config, BaseModel):
                    config = config.model_copy(update=kwargs)
                    # re-validate
                    config = config.model_validate(config.model_dump(warnings="none"))
                else:  # dataclass
                    config = replace(config, **kwargs)
        self.config: ConfigT = config
        self._config_type = type(self.config)
        self._configured = False
        self.on_init()

    def on_init(self) -> None:
        """Callback to be called when initializing"""

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)

    @classmethod
    def resolve_config_type(cls) -> Optional[Type]:
        return cls.get_annotation("config") or cls._generic_type

    @classmethod
    def get_annotation(cls, name: str) -> Optional[Type]:
        return getattr(cls, "__annotations__", {}).get(name, None)

    def dump(self, mode: Literal["python", "json"] = "python") -> Dict[str, Any]:
        """Dump the config to a dictionary.
        Args:
            mode: The mode to dump the config. "python" for standard dict, "json" (only applicable when the config is the `BaseModel` of pydantic) for JSON serializable dict.
        Returns:
            A dictionary representing the config with a "_target_" key indicating the class path.
        """
        if isinstance(self.config, BaseModel):
            config = self.config.model_dump(mode=mode)
        else:  # dataclass
            config = asdict(self.config)
        cls = self.__class__
        config["_target_"] = f"{cls.__module__}.{cls.__qualname__}"
        return config

    def save_config(
        self, path: Union[str, Path], mode: Literal["yaml", "json", "pickle"] = "yaml"
    ) -> None:
        """Save the config to a yaml file.
        Args:
            path: The file path to save the config.
            mode: The mode to save the config. "yaml" or "json".
        """
        with open(path, "w") as f:
            if mode == "yaml":
                if isinstance(self.config, BaseModel):
                    to_yaml_file(f, self.config, add_comments=True)
                else:  # dataclass
                    yaml.dump(asdict(self.config), f)
            elif mode == "json":
                json.dump(self.dump(mode="json"), f, indent=4)
            else:
                pickle.dump(self.config, f)

    @classmethod
    def load_from_config_file(cls, path: Union[str, Path]) -> None:
        """Load the config from a yaml or json file.
        Args:
            path: The file path to load the config from.
        """
        path = Path(path)
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                config = pickle.load(f)
                return cls(config)
        else:  # yaml or json
            config_type = cls.resolve_config_type()
            if not config_type or is_dataclass(config_type):
                with open(path, "r") as f:
                    config_dict = yaml.safe_load(f)
                    return cls(**config_dict)
            elif issubclass(config_type, BaseModel):
                config = parse_yaml_file_as(config_type, path)
                return cls(config)
            raise ValueError(
                "`config` must be annotated as a pydantic BaseModel or a dataclass."
            )

    @property
    def config_type(self) -> Optional[Type[ConfigT]]:
        """Get the config type."""
        return self._config_type

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        generic_type = resolve_generic_type(cls, InitConfigMixin)
        generic_type = (
            generic_type.__bound__
            if isinstance(generic_type, TypeVar)
            else generic_type
        )
        if generic_type:
            cls.get_logger().debug(
                f"Registered generic type for {cls.__name__}: {generic_type}"
            )
        cls._generic_type = generic_type


class ConfigurableBasis(ABC, InitConfigMixin[ConfigT]):
    @final
    def configure(self) -> bool:
        if self._configured:
            raise RuntimeError("Already configured")
        itf_type = self.get_annotation("interface")
        self.interface = None if itf_type is None else self._create_interface(itf_type)
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


def dump_or_repr(obj: Union[Any, ConfigurableBasis]) -> Union[Dict[str, Any], str]:
    """Dump the config if the object is a ConfigurableBasis, otherwise return the repr.
    Args:
        obj: The object to dump or repr.
    Returns:
        A dictionary representing the config or the repr string.
    """
    if callable(getattr(obj, "dump", None)):
        return obj.dump()
    else:
        config = getattr(obj, "config", None)
        if config is not None:
            if isinstance(config, BaseModel):
                return config.model_dump()
            elif is_dataclass(config):
                return asdict(config)
        return repr(obj)


if __name__ == "__main__":
    import logging
    from pydantic import ConfigDict

    logging.basicConfig(level=logging.INFO)

    class MyConfig(BaseModel):
        """Example configuration model."""

        param1: int = 10
        param2: str = "default"

    class MyComponent(ConfigurableBasis[MyConfig]):
        def on_configure(self) -> bool:
            self.get_logger().info(f"Configuring with {self.config}")
            return True

    class MyCompsConfig(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        comps: list[MyComponent]

    comp = MyComponent(param1=20)
    comp.configure()
    print(comp.dump())

    comps_config = MyCompsConfig(comps=[comp])
    print(comps_config)
    print(comps_config.model_dump(mode="json", fallback=lambda x: x.dump()))
