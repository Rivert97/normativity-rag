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
        if self._args.type == 'csv':
            self.__process_csv_file(self._args.file)
        elif self._args.type == 'txt':
            self.__process_txt_file(self._args.file)
        else:
            raise CLIException("Tipo de archivo no valido")

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Creates a vectorized database of data extracted from PDF files',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('-a', '--action', default='structure', choices=['structure', 'tree', 'save'], type=str, help='Action to perform')
        parser.add_argument('-c', '--collection-name', default='', help='When action is "create", set the name of the collection')
        parser.add_argument('-f', '--file', default='', type=str, help='Path to file containing the data or text of the document')
        parser.add_argument('-p', '--page', type=int, help='Number of page to be processed')
        parser.add_argument('-t', '--type', default='txt', choices=['txt', 'csv'], type=str, help='Type of input')
        parser.add_argument('--version', action='store_true', help='Show version of this tool')

        args = parser.parse_args()

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"File '{args.file} not found")

        if args.action == 'save' and args.collection_name == '':
            raise CLIException(f"Please specify the name of the colleciton")

        return args

    def __process_csv_file(self, filename: str):
        document_data = PdfDocumentData()
        document_data.load_data(filename)
        if self._args.page != None:
            splitter = NormativitySplitter(document_data.get_page_data(self._args.page, remove_headers=True))
        else:
            splitter = NormativitySplitter(document_data.get_data(remove_headers=True))

        splitter.analyze()

        if self._args.action == 'structure':
            splitter.show_file_structure()
        elif self._args.action == 'tree':
            splitter.show_tree()
        elif self._args.action == 'save':
            splits = splitter.extract_documents()

            storage = Storage()
            storage.save_documents(self._args.collection_name, splits)

    def __process_txt_file(self, filename: str):
        pass


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