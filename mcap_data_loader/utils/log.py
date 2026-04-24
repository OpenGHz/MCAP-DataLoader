import logging
from typing import Optional


_job_id_prefix = ""
_original_record_factory = logging.getLogRecordFactory()


def _job_id_record_factory(*args, **kwargs):
    record = _original_record_factory(*args, **kwargs)
    record.job_id = _job_id_prefix
    return record


logging.setLogRecordFactory(_job_id_record_factory)


def set_log_job_id(job_id: Optional[int], prefix: str = "") -> None:
    global _job_id_prefix
    _job_id_prefix = f"[{prefix}{job_id}] " if job_id is not None else ""


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "[%(levelname)s] %(asctime)s %(job_id)s%(name)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    ch = logging.StreamHandler()
    # ch.setLevel(level)
    ch.setFormatter(ColorfulFormatter())
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.addHandler(ch)
