class CLIException(Exception):
    def __init__(self, message):
        super().__init__(f"ERROR: {message}")
