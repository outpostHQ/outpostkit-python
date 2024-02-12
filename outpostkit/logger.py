import logging
from typing import Union


def init_outpost_logger(level: Union[str, int]):
    # Use the same settings as above for root logger
    logging.basicConfig(format="%(asctime)s %(message)s")
    outpost_logger = logging.getLogger("outpost_logger")
    outpost_logger.setLevel(logging.getLevelName(level))
    return outpost_logger
