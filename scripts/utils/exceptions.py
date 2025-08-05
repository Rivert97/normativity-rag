"""Module to define custom exceptions for the package."""

class CLIException(Exception):
    """Command Line Interface Exception."""
    def __init__(self, message):
        super().__init__(f"ERROR: {message}")
