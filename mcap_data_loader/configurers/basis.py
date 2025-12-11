from abc import ABC, abstractmethod
from typing import Type, TypeVar, Generic, final, Optional, Union
from collections.abc import Callable
from pathlib import Path
from argparse import ArgumentParser
from mcap_data_loader.utils.basic import import_string
import logging
import __main__
import os


T = TypeVar("T")


class ConfigurerBasis(ABC, Generic[T]):
    """The basis class for configurers (config backends)."""

    def __init__(self, config_class: Type[T]) -> None:
        self.config_class = config_class
        self._main_dir = Path(__main__.__file__).parent.resolve()
        self._main = None

    @abstractmethod
    def parse(self, config_path: Optional[str] = None, **kwargs) -> None:
        """Parse the command line arguments.
        Args:
            config_path (Optional[str]): The path to the configuration file.
            **kwargs: Additional keyword arguments.
        """

    @abstractmethod
    def on_configure(self) -> Union[int, T]:
        """The internal configure function to be implemented by subclasses.
        Returns:
            T: The configured instance.
        """

    @final
    def configure(self, main: Optional[Callable[[T], int]] = None) -> Union[int, T]:
        """Configure the given config class and return the instance.
        Args:
            The main function to be called with the configured instance.
        Returns:
            int: The return value of the main function.
        """
        self._main = main
        result = self.on_configure()
        if main is None and not isinstance(result, self.config_class):
            raise TypeError(
                f"The return value must be of type {self.config_class} when main is None, but got {type(result)}"
            )
        if main is not None and not isinstance(result, int):
            raise TypeError(
                f"The return value must be of type int when main is not None, but got {type(result)}"
            )
        return result

    @final
    def _check_config(self, config: T) -> T:
        if not isinstance(config, self.config_class):
            raise TypeError(
                f"The configured instance must be of type {self.config_class}, but got {type(config)}"
            )
        return config

    @final
    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the logger for the configurer.
        Returns:
            logging.Logger: The logger for the configurer.
        """
        return logging.getLogger(cls.__name__)


def main_argparse(
    prog: Optional[str] = None, default: Optional[str] = None
) -> Type[ConfigurerBasis]:
    parser = ArgumentParser(prog, add_help=False)
    parser.add_argument(
        "--configurer",
        "-cfger",
        default=os.environ.get(
            "AIRDC_CONFIGURER",
            default or "mcap_data_loader.configurers.hydra_cfger.Configurer",
        ),
        type=import_string,
        help="The configurer (config backend) name or package path",
    )
    parser.add_argument(
        "--main-help", action="store_true", help="Show this help message"
    )
    args, _ = parser.parse_known_args()
    if args.main_help:
        help_lines = parser.format_help().splitlines()
        help_lines[0] += " [CONFIGER_OPTIONS...]"
        print("\n".join(help_lines))
        exit(0)
    return args.configurer
