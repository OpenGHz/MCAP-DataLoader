from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from mcap_data_loader.configurers.basis import ConfigurerBasis, T
from mcap_data_loader.utils.file import find_file_paths
from mcap_data_loader.utils.basic import import_string
from hydra_zen import instantiate, store
from hydra.core import hydra_config
import hydra
import argparse
import sys
import os


os.environ["HYDRA_FULL_ERROR"] = "1"


class Configurer(ConfigurerBasis[T]):
    """The configurer using Hydra as the backend."""

    def parse(self, config_path=None) -> None:
        cwd = Path.cwd()
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config-path", "--path", default=config_path)
        parser.add_argument(
            "--base-dir",
            default=str(cwd),
            help="The base directory for config path."
            "__main__ for main file directory. Default to the current working directory.",
        )
        parser.add_argument(
            "--cfger-help", action="store_true", help="Show this help message"
        )
        parser.add_argument(
            "--add-cwd-mode",
            type=str,
            default="append",
            choices=["prepend", "append", "none"],
            help="Whether to add the current working directory to sys.path, and where to add it",
        )
        parser.add_argument(
            "--show-resolved",
            "-sr",
            action="store_true",
            help="Show the resolved config and exit",
        )
        args, unknown = parser.parse_known_args()
        self._show_resolved = args.show_resolved
        sys.argv = sys.argv[:1] + unknown
        config_name = "class_config"
        store(self.config_class, name=config_name)
        store.add_to_hydra_store()
        config_path = args.config_path
        if config_path is None:
            for path in find_file_paths(cwd, "config.yaml", 2):
                config_path = path
                break
            else:
                # TODO: should allow not having a config file?
                raise FileNotFoundError(
                    "The `config.yaml` file cannot be found automatically. Please ensure that the configuration file is placed in the working directory or at most one of its next-level subdirectories, or manually specify its path."
                )
        elif isinstance(config_path, str):
            if not config_path.startswith("pkg://"):
                config_path = Path(config_path)
        if isinstance(config_path, Path):
            # empty config_path means using the base_dir as config_path
            base_dir = self._main_dir if args.base_dir == "__main__" else args.base_dir
            if config_path.suffix == ".yaml":
                config_dir = config_path.parent
                config_name = config_path.stem
            else:
                config_dir = config_path
            if not config_dir.is_absolute():
                config_dir = (base_dir / config_dir).absolute()
            if not config_dir.exists():
                raise FileNotFoundError(f"{config_dir} not found")
            print(f"Base dir: {base_dir}")
            config_path = str(config_dir.absolute())
            print(f"Using config path: {config_path}")
        self._dict_config = None
        add_cwd_mode = args.add_cwd_mode
        cwd = str(Path.cwd().absolute())
        # syspath = sys.path.copy()
        if add_cwd_mode == "prepend":
            sys.path.insert(0, cwd)
        elif add_cwd_mode == "append":
            sys.path.append(cwd)
        self._config_path = config_path
        self._config_name = config_name

    @staticmethod
    def merge_dicts(base: dict, overrides: dict):
        return OmegaConf.merge(base, overrides)

    @staticmethod
    def instantiate(config, overrides: dict = None):
        return instantiate(config, **(overrides or {}))

    def __set_dict_config(self, dict_config: DictConfig) -> None:
        self._dict_config = dict_config
        self.get_logger().info(
            f"Original working directory : {hydra.utils.get_original_cwd()}"
        )
        self.get_logger().info(
            f"Output directory  : {hydra_config.HydraConfig.get().runtime.output_dir}"
        )
        if self._show_resolved:
            OmegaConf.resolve(dict_config)
            print(OmegaConf.to_yaml(dict_config))
            exit(0)

    def __instantiate_config(self):
        config_instance = instantiate(self._dict_config)
        if isinstance(config_instance, self.config_class):
            return config_instance
        else:
            config_instance = OmegaConf.to_object(config_instance)
            if isinstance(config_instance, dict):
                return self.config_class(**config_instance)
            else:
                raise TypeError(
                    f"The instantiated config {config_instance} must be a `{self.config_class}` or a dict, but got `{type(config_instance)}`."
                )

    def __set_and_run(self, dict_config: DictConfig) -> T:
        self.__set_dict_config(dict_config)
        return self._main(self._check_config(self.__instantiate_config()))

    def on_configure(self) -> T:
        hydra_main = hydra.main(self._config_path, self._config_name, None)
        if self._main is None:
            hydra_main(self.__set_dict_config)()
            # NOTE: restoring sys.path may cause issues if using multiprocessing with spawn method
            # sys.path = syspath
            if self._dict_config is None:
                exit(0)
            return self.__instantiate_config()
        else:
            return hydra_main(self.__set_and_run)()


OmegaConf.register_new_resolver("merge_cfg", Configurer.merge_dicts)
OmegaConf.register_new_resolver("instantiate", Configurer.instantiate)
OmegaConf.register_new_resolver("import_string", import_string)


if __name__ == "__main__":
    from pydantic import BaseModel

    class DataCollectionArgs(BaseModel):
        option: str = "foo"

    configurer = Configurer(
        DataCollectionArgs,
    )
    configurer.parse()
    assert configurer.configure()
