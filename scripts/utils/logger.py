"""Module to handle logging gracefully."""

import logging
import sys

class AppLogger():
    """Class to handle the application logger."""

    @classmethod
    def setup_root_logger(cls, level:int|str=logging.INFO, log_file:str=None, log_console:int=0):
        """Setup the root logger with the desired parameters.

        This settings affect all children loggers.
        """
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        if log_console != 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

    @classmethod
    def get_logger(cls, name:str=__name__) -> logging.Logger:
        """Get a logger by its name."""
        logger = logging.getLogger(name)
        return logger
