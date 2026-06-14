import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file="logs/app.log", max_bytes=5 * 1024 * 1024, backup_count=5):
    """
    Sets up a rotating file logger.
    - log_file: Path to the log file.
    - max_bytes: Max size of a single log file before it's rotated (default 5MB).
    - backup_count: Number of old log files to keep.
    """
    # if the setup script is in 'logs/', we want app.log in the same 'logs/' dir.
    # when running from the root, 'logs/app.log' is correct.

    logger = logging.getLogger("BrainTumorDetection")
    logger.setLevel(logging.INFO)

    # prevent multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # create rotating file handler
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # also add a console handler for visibility during development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# pre-initialize the logger
logger = setup_logging()

