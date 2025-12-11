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
    List,
    final,
    get_type_hints,
)
from collections.abc import Mapping, Callable
from typing_extensions import get_args, Self
from pydantic import BaseModel, TypeAdapter
from pydantic_yaml import to_yaml_file
from pathlib import Path
from functools import cache
from weakref import WeakSet
from mcap_data_loader.utils.basic import (
    DataClassProto,
    import_string,
    get_fully_qualified_class_name,
)
from copy import copy, deepcopy
from toolz.dicttoolz import get_in
import inspect
import yaml
import json
import pickle


class NoConfig(BaseModel, frozen=True):
    """A placeholder config class indicating no configuration is needed."""


OtherCfgType = Optional[
    Union[Type[Union[DataClassProto, BaseModel]], Dict[str, Any], str, Path]
]
ConfigType = Union[BaseModel, DataClassProto]
AllConfigType = Union[ConfigType, OtherCfgType]


class InitConfigMeta(type):
    def __call__(cls, config: AllConfigType = None, **kwargs):
        error_msg = """
        If not provided as an arg, `config` must be annotated as a subclass of `ConfigType` 
        either in the `__init__` method or at the class scope. If a class truly does not 
        require any configuration, you can mark the type as `None` at the class scope."""
        if isinstance(config, (str, Path)):
            config = cls.config_from_file(config)
        config_type = (
            config
            if isinstance(config, type)
            else cls.resolve_config_type(cls)
            if (config is None or isinstance(config, Mapping))
            else type(config)
        )
        cls_path = kwargs.pop("_target_", None)
        if config_type is None:
            # try to get from _target_
            if cls_path is not None:
                config_type = import_string(cls_path)
        # TODO: it is better to support more flexible config types
        # e.g. dict, etc.
        if not cls._check_config_type(config_type):
            raise TypeError(f"{cls}" + error_msg + f" Got {config_type} instead.")
        if not issubclass(config_type, BaseModel):
            adapter = TypeAdapter(config_type)
        if isinstance(config, Mapping):
            # if (
            #     get_fully_qualified_class_name(config)
            #     == "omegaconf.dictconfig.DictConfig"
            # ):
            #     from omegaconf import OmegaConf

            #     config = OmegaConf.to_object(config)
            config.update(kwargs)
            config = config_type(**config)
        elif config is None or isinstance(config, type):
            if issubclass(config_type, BaseModel):
                config = config_type(**kwargs)
            else:
                config = adapter.validate_python(kwargs)
        elif kwargs:  # mainly used by yaml config, e.g. hydra
            if isinstance(config, BaseModel):
                config = config.model_copy(update=kwargs)
                # re-validate
                config = config.model_validate(config.model_dump(warnings="none"))
                extra = kwargs.keys() - config_type.model_fields.keys()
                if extra:
                    cls.get_logger().warning(
                        f"Extra fields {extra} found in config, which will be ignored."
                    )
            else:  # dataclass
                config = replace(config, **kwargs)
                config = adapter.validate_python(asdict(config))
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

    @staticmethod
    def _check_config_type(config_type: Type) -> bool:
        return isinstance(config_type, type) and issubclass(
            config_type, get_args(ConfigType)
        )

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)

    @staticmethod
    @cache
    def resolve_config_type(cls) -> Optional[Type]:
        cfg_type = InitConfigMeta.get_annotation(cls, "config")
        if cfg_type in (None, ...) or (getattr(cls, "config", object) in (None, ...)):
            cfg_type = NoConfig
        elif cfg_type is object:
            cfg_type = get_type_hints(cls.__init__).get("config", None)
        return cfg_type

    @staticmethod
    @cache
    def get_annotation(cls, name: str) -> Optional[Any]:
        return getattr(cls, "__annotations__", {}).get(name, object)

    @staticmethod
    def config_from_file(path: Union[str, Path]) -> AllConfigType:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} not found.")
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                config = pickle.load(f)
        else:  # yaml or json
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        return config


class InitConfigABCMeta(InitConfigMeta, ABCMeta):
    """A metaclass that combines InitConfigMeta and ABCMeta."""


class InitConfigMixinBasis:
    """A mixin class for initializing with a config."""

    def __init__(self, config: ConfigType) -> None:
        """Initialize with a config. It is only used for type hinting in the current class;
        subclasses can override it at will. Note that you should try to avoid directly modifying
        the config variable internally, as this may lead to potential problems, such as inconsistent
        behavior when the same config variable is used to instantiate the class multiple times. Unless
        there is a special need, such as a class that might be specifically designed to modify external
        configurations, it is best to update the copied variables and then reassign them to the instance.
        Furthermore, for maximum security and faster value retrieval, the configuration class should be
        set to frozen unless there is a clear need for modification. Conversely, due to this security
        mechanism, in complex configuration chains, the original external configuration may not accurately
        reflect the internal configuration of the instance. Therefore, when a configuration needs to be
        referenced externally, this consistency issue should be considered. However, this should not be
        seen as a drawback of this security mechanism, as it is an issue that must be considered in all
        circumstances. On the contrary, it forces developers to make more proactive and reasonable designs.
        It is undeniable that shallow copy operations incur additional time consumption, but this is usually
        negligible."""
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
        self,
        path: Union[str, Path],
        mode: Optional[Literal["yaml", "json", "pickle"]] = None,
    ) -> None:
        """Save the config to a yaml file.
        Args:
            path: The file path to save the config.
            mode: The mode to save the config. If None, it will be inferred from the file extension.
        """
        with open(path, "w") as f:
            mode = Path(path).suffix.removeprefix(".") if mode is None else mode
            if mode in ("yaml", "yml"):
                if isinstance(self.config, BaseModel):
                    to_yaml_file(f, self.config, add_comments=True)
                else:  # dataclass
                    yaml.dump(asdict(self.config), f)
            elif mode == "json":
                json.dump(self.dump(mode="json"), f, indent=4)
            elif mode in ("pickle", "pkl"):
                pickle.dump(self.config, f)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

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

    def __repr__(self):
        return f"{self.__module__}.{self.__class__.__qualname__}"


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


def dump_omegaconf(obj: Any) -> Dict[str, Any]:
    """Dump an omegaconf object to a dictionary.
    Args:
        obj: The omegaconf object to dump.
    Returns:
        A dictionary representing the omegaconf object.
    Raises:
        TypeError: If the object is not an omegaconf object.
    """
    from omegaconf import OmegaConf

    if get_fully_qualified_class_name(obj) in {
        "omegaconf.dictconfig.DictConfig",
        "omegaconf.listconfig.ListConfig",
    }:
        return OmegaConf.to_object(obj)
    raise TypeError("Not an omegaconf object.")


def dump_or_repr(
    obj: Union[Any, ConfigurableBasis], handler: Optional[Callable] = None
) -> Union[Dict[str, Any], str]:
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
    try:
        return dump_omegaconf(obj)
    except TypeError:
        if handler is not None:
            return handler(obj)
        return repr(obj)


def fetch_config(
    target: Union[Path, str, Mapping],
    keys: Union[List[str], str] = "",
    default=None,
    no_default: bool = True,
) -> Any:
    """Fetch the config from a target mapping or a yaml file.
    Args:
        target: The target mapping or file path.
        keys: The keys or a dot-separated key to the config. If empty, return the whole config.
        default: The default value if the key is not found.
        no_default: Whether to raise an error if the key is not found and no default is provided.
    Returns:
        The config object.
    """
    if isinstance(target, (str, Path)):
        with open(target, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = target
    if not keys:
        return data
    if isinstance(keys, str):
        keys = keys.split(".")
    return get_in(keys, data, default, no_default)


if __name__ == "__main__":
    config = {"a": {"b": 1}}
    assert fetch_config(config, "a.b") == 1
    fetch_config(config, "a.c", 2, False) == 2
    try:
        fetch_config(config, "a.c")
    except KeyError:
        pass
