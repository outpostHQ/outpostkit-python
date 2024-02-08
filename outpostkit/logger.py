import logging
from typing import Union


def init_outpost_logger(level: Union[str, int]):
    # Use the same settings as above for root logger
    outpost_logger = logging.getLogger("outpost_logger")
    outpost_logger.setLevel(logging.getLevelName(level))
    outpost_logger.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
    return outpost_logger
