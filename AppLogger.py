import logging
import sys

class AppLogger():

    @classmethod
    def setup_root_logger(cls, level:int|str=logging.INFO, log_file:str=None, log_console:int=0):
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
        logger = logging.getLogger(name)
        return logger