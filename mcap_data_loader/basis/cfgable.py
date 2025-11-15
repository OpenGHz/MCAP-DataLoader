from abc import ABCMeta, abstractmethod
from dataclasses import asdict, replace, is_dataclass
from logging import getLogger
from typing import (
    Union,
    Optional,
    Type,
    Dict,
    Any,
    Literal,
    Set,
    final,
    get_type_hints,
)
from typing_extensions import get_args, Self
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from pathlib import Path
from functools import cache
from weakref import WeakSet
from mcap_data_loader.utils.basic import get_class_type, DataClassProto
from copy import copy, deepcopy
import inspect
import yaml
import json
import pickle


class NoConfig:
    """A placeholder config class indicating no configuration is needed."""


OtherCfgType = Optional[Union[Type[Union[DataClassProto, BaseModel]], Dict[str, Any]]]
ConfigType = Union[BaseModel, DataClassProto, NoConfig]


class InitConfigMeta(type):
    def __call__(cls, config: Union[ConfigType, OtherCfgType] = None, **kwargs):
        error_msg = """
        `config` must be annotated as a subclass of `ConfigType` in the `__init__`
        method or at the top level class if not provided as an arg. If a subclass 
        truly does not require any configuration, you can pass an arbitrary (not None) 
        instance parameter during instantiation, typically `...`, or you can mark the 
        type as `NoConfig` or `...`."""
        # mainly used by yaml config, e.g. hydra
        if config is None or isinstance(config, type):
            config_type = config or cls.resolve_config_type(cls)
            if config_type is ...:
                config_type = NoConfig
            if not (
                isinstance(config_type, type)
                and issubclass(config_type, get_args(ConfigType))
            ):
                cls_path = kwargs.pop("_target_", None)
                if cls_path is not None:
                    config_type = get_class_type(cls_path)
                raise ValueError(error_msg + f" Got type of {config_type}.")
            # self.get_logger().info(f"{kwargs}")
            config = config_type(**kwargs)
            # check pydantic extra kwargs
            if isinstance(config, BaseModel):
                extra = kwargs.keys() - config.__class__.model_fields.keys()
                if extra:
                    cls.get_logger().warning(
                        f"Extra fields {extra} found in config, which will be ignored."
                    )
        else:  # mainly used by instancing manually
            if isinstance(config, dict):
                config_type = cls.resolve_config_type(cls)
                if not config_type:
                    cls_path = config.pop("_target_", None)
                    if cls_path is not None:
                        config_type = get_class_type(cls_path)
                    raise ValueError(error_msg)
                config = config_type(**config)
            if kwargs:  # rarely used
                if isinstance(config, BaseModel):
                    config = config.model_copy(update=kwargs)
                    # re-validate
                    config = config.model_validate(config.model_dump(warnings="none"))
                else:  # dataclass
                    config = replace(config, **kwargs)
        # NOTE: since there may be arbitrary instances in the config,
        # we should not deep copy the config here to avoid potential issues.
        # if isinstance(config, BaseModel):
        #     cfg_copy = config.model_copy(deep=True)
        # else:
        #     cfg_copy = deepcopy(config)
        instance: "InitConfigMixinBasis" = super().__call__(config)
        # instance.__config = cfg_copy
        if not hasattr(instance, "config"):
            instance.config = config
        instance.config_post_init()
        return instance

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)

    @staticmethod
    @cache
    def resolve_config_type(cls) -> Optional[Type]:
        return InitConfigMeta.get_annotation(cls, "config") or get_type_hints(
            cls.__init__
        ).get("config", None)

    @staticmethod
    @cache
    def get_annotation(cls, name: str) -> Optional[Type]:
        return getattr(cls, "__annotations__", {}).get(name, None)


class InitConfigABCMeta(InitConfigMeta, ABCMeta):
    """A metaclass that combines InitConfigMeta and ABCMeta."""


class InitConfigMixinBasis:
    def __init__(self, config: ConfigType) -> None:
        """Initialize with a config. It is only used for type hinting in the current class;
        subclasses can override it at will. Note that you should try to avoid directly modifying
        the config variable internally, as this may lead to potential problems, such as inconsistent
        behavior when the same config variable is used to instantiate the class multiple times."""
        self.config = config

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)

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
            config_type = InitConfigMeta.resolve_config_type()
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

    def config_post_init(self) -> None:
        """A hook to be called after the config is set."""

    @final
    def copy(self, deep: bool = False) -> Self:
        """Create a copy of the current instance with the same config."""
        if isinstance(self.config, BaseModel):
            config = self.config.model_copy(deep=deep)
        else:
            method = copy if not deep else deepcopy
            config = method(self.config)
        return self.__class__(config)


class InitConfigMixin(InitConfigMixinBasis, metaclass=InitConfigMeta):
    """Mixin class for initializing with a config."""


class InitConfigABCMixin(InitConfigMixinBasis, metaclass=InitConfigABCMeta):
    """Mixin class for initializing with a config and supporting abstract methods."""


class ConfigurableBasis(InitConfigABCMixin):
    """A basis class for easy configuring."""

    __instances: Set[Self] = WeakSet()

    def config_post_init(self):
        self._configured = False
        self.__class__.__instances.add(self)

    @final
    def configure(self) -> bool:
        if self._configured:
            raise RuntimeError("Already configured")
        itf_type = InitConfigMeta.get_annotation(self.__class__, "interface")
        self.interface = self._create_interface(itf_type)
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

    @classmethod
    def all_configure(cls) -> bool:
        """Configure all instances of this class and its subclasses."""
        for instance in cls.__instances:
            if not instance.configured:
                if not instance.configure():
                    cls.get_logger().error(f"Failed to configure instance: {instance}")
                    return False
        return True

    def _create_interface(self, class_type: Optional[Type]):
        """Create the interface instance based on the config and the class type annotation.
        The subclasses can override this method if needed.
        """
        if class_type is None:
            return None
        sig = inspect.signature(class_type)
        if "config" in sig.parameters.keys():
            interface = class_type(config=self.config)
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
            interface = class_type(**{key: cfg_dict[key] for key in com_keys})
        return interface


def dump_or_repr(obj: Union[Any, ConfigurableBasis]) -> Union[Dict[str, Any], str]:
    """Dump the config if the object is a ConfigurableBasis, otherwise return the repr.
    Args:
        obj: The object to dump or repr.
    Returns:
        A dictionary representing the config or the repr string.
    """
    if callable(getattr(obj, "dump", None)):
        try:
            return obj.dump()
        except Exception as e:
            getLogger(f"{dump_or_repr.__name__}").error(f"Failed to dump config: {e}")
    config = getattr(obj, "config", None)
    if config is not None:
        if isinstance(config, BaseModel):
            return config.model_dump()
        elif is_dataclass(config):
            return asdict(config)
        elif isinstance(config, dict):
            return config
    return repr(obj)


if __name__ == "__main__":
    import logging
    from pydantic import ConfigDict

    logging.basicConfig(level=logging.INFO)

    class MyConfig(BaseModel):
        """Example configuration model."""

        param1: int = 10
        param2: str = "default"

    class MyComponent(ConfigurableBasis):
        def __init__(self, config: MyConfig):
            self.config = config

        def on_configure(self) -> bool:
            self.get_logger().info(f"Configuring with {self.config}")
            return True

    class MyCompsConfig(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        comps: list[MyComponent]

    comp = MyComponent(param1=20)
    assert comp.configure()
    assert comp.all_configure()
    print(comp.dump())
    comp.copy()
    comp.copy(True)
    comps_config = MyCompsConfig(comps=[comp])
    print(comps_config)
    print(comps_config.model_dump(mode="json", fallback=lambda x: x.dump()))
