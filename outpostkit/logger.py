import logging
import os
from typing import Union


def init_outpost_logger(
    name, level: Union[str, int] = os.getenv("LOGLEVEL", "INFO").upper()
) -> logging.Logger:
    # Use the same settings as above for root logger
    logging.basicConfig(format="%(asctime)s %(message)s")
    outpost_logger = logging.getLogger(name)
    outpost_logger.setLevel(logging.getLevelName(level))
    return outpost_logger
