import logging
import os

log_file_path = os.path.expanduser("/var/log/outpost/outpostkit.log")
outpost_folder = os.path.dirname(log_file_path)
if not os.path.exists(outpost_folder):
    # Create the ~/.outpost folder if it doesn't exist
    os.makedirs(outpost_folder)


def create_lfs_logger(name: str):
    _log = logging.getLogger(name)
    _log.handlers.clear()
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    _log.addHandler(file_handler)
    return _log
