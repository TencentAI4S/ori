# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import functools
import logging
import os
import sys

from fvcore.common.file_io import PathManager
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def get_logger(name="tgnn"):
    return logging.getLogger(name)


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name: str = "tgnn",
                 output: str = None,
                 color: bool = True,
                 distributed_rank: int = 0,
                 abbrev_name: str = None):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        name (str): the root module name of this logger
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout logging: single machine, logging in master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if output is not None:
            output = str(output)
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, f"{name}.log")

            if distributed_rank > 0:
                filename = filename + ".rank{}".format(distributed_rank)

            PathManager.mkdirs(os.path.dirname(filename))

            fh = logging.StreamHandler(_cached_log_stream(filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")
