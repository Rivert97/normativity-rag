"""Module to define base clases for various functions."""

import argparse
import os
import sys

import dotenv

from .logger import AppLogger
from .exceptions import CLIException

dotenv.load_dotenv()

class CLI:
    """Base class for CLI scripts that handle logging."""

    def __init__(self, name:str, description:str, version:int):
        self.name = name
        self.description = description
        self.version = version

        try:
            AppLogger.setup_root_logger(
                int(os.getenv('LOG_LEVEL', '20')),
                os.getenv('LOG_FILE', f"{self.name}.log"),
                int(os.getenv('LOG_CONSOLE', '0'))
            )
        except ValueError as e:
            print(f"Environment variables file is corrupted: {e}")
            sys.exit(1)

        self._logger = AppLogger.get_logger(self.name)
        self._logger.info(' '.join(sys.argv))

        self.parser = argparse.ArgumentParser(
            prog=self.name,
            description=self.description,
            epilog=f'%(prog)s-{self.version}, Roberto Garcia <r.garciaguzman@ugto.mx>',
            formatter_class=argparse.RawDescriptionHelpFormatter)

    def process_args(self):
        """Add and process the arguments of the script."""
        self.parser.add_argument('-v', '--version', action='version', version=self.version)

    def run(self):
        """Run the script logic."""

    def get_logger(self):
        """Get the logger handler."""
        return self._logger

def run_cli(cli_class: CLI):
    """Run a CLI class."""
    try:
        controller = cli_class()
        controller.process_args()
        controller.run()
    except CLIException as e:
        print(e)
        controller.get_logger().error(e)
        sys.exit(1)
