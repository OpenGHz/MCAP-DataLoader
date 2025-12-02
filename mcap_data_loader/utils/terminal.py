from termcolor import colored
from typing import Optional, Union
import sys
import logging


class Bcolors:
    """Utility class for coloring terminal text."""

    @staticmethod
    def green(text: str) -> str:
        return colored(text, "green")

    @staticmethod
    def blue(text: str) -> str:
        return colored(text, "blue")

    @staticmethod
    def magenta(text: str) -> str:
        return colored(text, "magenta")

    @staticmethod
    def cyan(text: str) -> str:
        return colored(text, "cyan")


def is_terminal_stream(stream) -> bool:
    """Check if the given stream is a terminal (TTY) stream."""
    if stream in (sys.stdout, sys.stderr):
        return True
    try:
        return stream.isatty()  # 终端设备通常支持 isatty()
    except (AttributeError, OSError):
        return False


def get_terminal_formatter(logger: logging.Logger) -> Optional[logging.Formatter]:
    """Get the formatter for terminal stream handlers in the given logger."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if is_terminal_stream(handler.stream):
                return handler.formatter
    return None


def get_effective_terminal_formatter(
    logger: logging.Logger,
) -> Optional[logging.Formatter]:
    """Get the effective terminal formatter for the given logger, considering propagation."""
    current = logger
    while current:
        fmt = get_terminal_formatter(current)
        if fmt is not None:
            return fmt
        if not current.propagate:
            break
        if current.parent:
            current = current.parent
        else:
            break
    return None


def format_log_message(
    fmt: Optional[Union[str, logging.Formatter]],
    name,
    message,
    level=logging.INFO,
    args=(),
):
    """Format a log message using the given format, level, name, message, and args."""
    if fmt is None:
        return message
    formatter = logging.Formatter(fmt) if isinstance(fmt, str) else fmt
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=args,
        exc_info=None,
    )
    return formatter.format(record)


class FormatTerminalMessage:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.formatter = get_effective_terminal_formatter(logger)

    def format(
        self,
        message: str,
        level: int = logging.INFO,
        args: tuple = (),
    ) -> str:
        return format_log_message(
            self.formatter, self.logger.name, message, level, args
        )


if __name__ == "__main__":
    from logging import getLogger, basicConfig

    basicConfig(level=logging.INFO)
    logging._nameToLevel
    logging._levelToName

    logger = getLogger("terminal_test")
    logger.info("This is normal")
    logger.info(Bcolors.green("This is green"))
    logger.info(Bcolors.blue("This is blue"))
    logger.info(Bcolors.magenta("This is magenta"))
    logger.info(Bcolors.cyan("This is cyan"))
    logger.info("End.")

    formatter = get_effective_terminal_formatter(logger)
    print("Effective formatter:", formatter)

    msg = format_log_message(
        formatter, logger.name, level=logging.ERROR, message="Something went wrong!"
    )
    print(msg)  # [ERROR] Something went wrong!

    ftm = FormatTerminalMessage(logger)
    print(
        ftm.format(
            "Formatted message via FormatTerminalMessage class.", logging.WARNING
        )
    )
