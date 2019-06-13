from __future__ import absolute_import, division, print_function

import os
import sys
import logging

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG


def get_logger(filepath, level=INFO, stream=sys.stderr):
    name = os.path.splitext(os.path.basename(filepath))[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger
