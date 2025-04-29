import argparse
import os
import glob
import dotenv

from Loaders import PdfMixedLoader
from AppLogger import AppLogger

dotenv.load_dotenv()

PROGRAM_NAME = 'NormativityRAG'
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
        if self._args.file != '':
            self.__process_file()
        elif self._args.directory != '':
            self.__process_directory()
        else:
            raise CLIException("Input not specified")

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Creates a vectorized database of PDF files',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('--cache-dir', default='./.cache/', type=str, help='Directory to be used as cache. Defaults to ./.cache/')
        parser.add_argument('-d', '--directory', default='', type=str, help='Directory to be processed in directory mode')
        parser.add_argument('-f', '--file', default='', type=str, help='File to be processed in single file mode')
        parser.add_argument('-o', '--output', default='./', type=str, help='File or directory to store the output text file(s). When -d is used, this defaults to ./')
        parser.add_argument('-p', '--page', type=int, help='Number of page to be processed')
        parser.add_argument('--version', action='store_true', help='Show version of this tool')

        args = parser.parse_args()

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"Input file '{args.file}' not found")

        if args.directory != '' and not os.path.exists(args.directory):
            raise CLIException(f"Input directory '{args.directory}' not found")

        if args.file == '' and args.directory == '':
            raise CLIException("Please specify an input file or directory")

        if not os.path.exists('/'.join(args.cache_dir.split('/')[:-1])):
            raise CLIException("Parent cache directory must exist")

        if args.directory != '':
            if not os.path.exists(args.output):
                raise CLIException("Destination directory does not exist")

        if args.file != '':
            dirname = os.path.dirname(args.output)
            if dirname != '' and not os.path.exists(dirname):
                raise CLIException("Output path does not exist")

        return args

    def __process_file(self):
        pdf_loader = PdfMixedLoader(self._args.file, self._args.cache_dir)
        if self._args.page != None:
            text = pdf_loader.get_page_text(self._args.page)
        else:
            text = pdf_loader.get_text()

        if self._args.output == './':
            print(text)
        else:
            self.__save_text(text, self._args.output)

    def __process_directory(self):
        for file in glob.glob(f'{self._args.directory}/*.pdf'):
            basename = ''.join(os.path.basename(file).split('.')[:-1])

            pdf_loader = PdfMixedLoader(file, self._args.cache_dir)
            if self._args.page != None:
                text = pdf_loader.get_page_text(self._args.page)
                out_name = f"{self._args.output}/{basename}_{self._args.page}.txt"
            else:
                text = pdf_loader.get_text()
                out_name = f"{self._args.output}/{basename}.txt"

            self.__save_text(text, out_name)

    def __save_text(self, text:str, filename: str):
        try:
            with open(filename, 'w') as f:
                f.write(text)
        except FileNotFoundError as e:
            self._logger.error(e)
            print("Output directory not found")

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