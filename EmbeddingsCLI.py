import argparse
import os
import dotenv

from AppLogger import AppLogger
from PdfDocumentData import PdfDocumentData
from Splitters import NormativitySplitter
from Storage import Storage

dotenv.load_dotenv()

PROGRAM_NAME = 'EmbeddingsCLI'
VERSION = '1.00.00'

class CLIException(Exception):
    def __init__(self, message):
        super().__init__(f"{PROGRAM_NAME} ERROR: {message}")

class CLIController():
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        self._logger = AppLogger.get_logger('CLIController')

        self._args = self.__process_args()

    def run(self):
        self.__process_csv_file(self._args.csv_file)

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Creates a vectorized database of data extracted from PDF files',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('-c', '--csv-file', default='', type=str, help='Load data directly from a .csv file')
        parser.add_argument('-p', '--page', type=int, help='Number of page to be processed')
        parser.add_argument('--version', action='store_true', help='Show version of this tool')

        args = parser.parse_args()

        if args.csv_file != '' and not os.path.exists(args.csv_file):
            raise CLIException(f"CSV file '{args.csv_file}' not found")

        if args.csv_file == '':
            raise CLIException("Please specify an input file")

        return args

    def __process_csv_file(self, filename: str):
        document_data = PdfDocumentData()
        document_data.load_data(filename)
        if self._args.page != None:
            splitter = NormativitySplitter(document_data.get_page_data(self._args.page, remove_headers=True))
        else:
            splitter = NormativitySplitter(document_data.get_data(remove_headers=True))

        splitter.analyze()

        splits = splitter.extract_documents()

        document_name = os.path.basename(filename)
        storage = Storage()
        storage.save_documents(document_name, splits)

if __name__ == "__main__":
    try:
        AppLogger.setup_root_logger(
            int(os.getenv('LOG_LEVEL', '20')),
            os.getenv('LOG_FILE', f"{PROGRAM_NAME}.log"),
            int(os.getenv('LOG_CONSOLE', '0'))
        )
    except ValueError as e:
        print(f"Environment variables file is corrupted: {e}")
        exit(1)

    _logger = AppLogger.get_logger(PROGRAM_NAME)

    try:
        controller = CLIController()
        controller.run()
    except CLIException as e:
        print(e)
        _logger.error(e)
        exit(1)