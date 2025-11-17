from termcolor import colored


class Bcolors:
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


if __name__ == "__main__":
    from logging import getLogger

    logger = getLogger("terminal_test")
    logger.info("This is normal")
    logger.info(Bcolors.green("This is green"))
    logger.info(Bcolors.blue("This is blue"))
    logger.info(Bcolors.magenta("This is magenta"))
    logger.info(Bcolors.cyan("This is cyan"))
    logger.info("End.")
