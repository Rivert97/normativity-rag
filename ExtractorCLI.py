import argparse
import os
import glob
import dotenv
import sys

from document_loaders.pdf import PyPDFMixedLoader, PyPDFLoader, OCRLoader
from utils.logger import AppLogger
from utils.exceptions import CLIException

dotenv.load_dotenv()

PROGRAM_NAME = 'ExtractorCLI'
VERSION = '1.00.00'

class CLIController():
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        self._logger = AppLogger.get_logger(PROGRAM_NAME)

        self.print_to_console = True

        self._args = self.__process_args()

    def run(self):
        if self._args.file != '':
            self.__process_file(self._args.file, self._args.output)
        elif self._args.directory != '':
            self.__process_directory()
        else:
            raise CLIException("Input not specified")

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Loads a PDF file and converts it to plain text and retrieves the position data of each word',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('--cache-dir', default='./.cache', type=str, help='Directory to be used as cache. Defaults to ./.cache')
        parser.add_argument('-d', '--directory', default='', type=str, help='Directory to be processed in directory mode')
        parser.add_argument('-f', '--file', default='', type=str, help='File to be processed in single file mode')
        parser.add_argument('-k', '--keep-cache', default=False, action='store_true', help='Keep Tesseract cache after processing. Usefull when the same file is going to be processed multiple times')
        parser.add_argument('-l', '--loader', default='mixed', type=str, choices=['mixed', 'text', 'ocr'], help='Type of loader to use. Defaults to mixed')
        parser.add_argument('-o', '--output', default='', type=str, help='File or directory to store the output text file(s). When -d is used, this defaults to ./')
        parser.add_argument('-p', '--page', type=int, help='Number of page to be processed')
        parser.add_argument('-t', '--type', default='txt', choices=['txt', 'csv'], nargs='+', type=str, help='Type(s) of output(s). Defaults to txt')
        parser.add_argument('-v', '--version', action='version', version=VERSION)

        args = parser.parse_args()

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"Input file '{args.file}' not found")

        if args.directory != '' and not os.path.exists(args.directory):
            raise CLIException(f"Input directory '{args.directory}' not found")

        if args.file == '' and args.directory == '':
            raise CLIException("Please specify an input file or directory")

        args.cache_dir = args.cache_dir.rstrip('/')
        if not os.path.exists(os.path.split(args.cache_dir)[0]):
            raise CLIException("Parent cache directory must exist")

        if args.directory != '':
            if args.output == '':
                args.output = './'
            if not os.path.isdir(args.output):
                raise CLIException("Destination directory does not exist")

        if args.file != '':
            dirname = os.path.dirname(args.output)
            if dirname != '' and not os.path.exists(dirname):
                raise CLIException("Output path does not exist")

        if args.loader == 'text' and args.type == 'csv':
            raise CLIException("Type of output not supported for '{args.loader}' loader")

        if args.output != '':
            self.print_to_console = False

        return args

    def __process_file(self, filename: str, output: str = None):
        self._logger.info(f'Processing file {filename}')

        pdf_loader = self.__get_loader(filename)
        self.__make_output(pdf_loader, output)

        if not self._args.keep_cache:
            pdf_loader.clear_cache()

    def __process_directory(self):
        for file in glob.glob(f'{self._args.directory}/*.pdf'):
            basename = ''.join(os.path.basename(file).split('.')[:-1])
            if self._args.page != None:
                out_name = f"{self._args.output}/{basename}_{self._args.page}"
            else:
                out_name = f"{self._args.output}/{basename}"

            self.__process_file(file, out_name)

    def __get_loader(self, filename:str):
        self._logger.info(f'Using "{self._args.loader}" loader')

        if self._args.loader == 'mixed':
            loader = PyPDFMixedLoader(self._args.cache_dir)
            if self._args.page != None:
                self._logger.info(f'Processing page {self._args.page}')

                loader.load_page(filename, self._args.page)
            else:
                loader.load(filename)
        elif self._args.loader == 'text':
            loader = PyPDFLoader(filename)
        elif self._args.loader == 'ocr':
            loader = OCRLoader(filename, self._args.cache_dir)
        else:
            raise CLIException("Invalid type of loader")

        return loader

    def __make_output(self, pdf_loader: PyPDFMixedLoader, output:str=None):
        base_filename = os.path.splitext(output)[0]
        if 'txt' in self._args.type:
            self._logger.debug('Generating text output')

            if self._args.page is not None:
                text = pdf_loader.get_page_text(self._args.page)
            else:
                text = pdf_loader.get_text()

            if self.print_to_console:
                print(text)
            else:
                self.__save_text(text, base_filename + '.txt')
                self._logger.info(f'Text output saved to {base_filename}.txt')

        if 'csv' in self._args.type:
            self._logger.debug('Generating csv output')

            data = pdf_loader.get_document_data()

            if self.print_to_console:
                if self._args.page is not None:
                    print(data.get_page_data(self._args.page).to_csv(index=False))
                else:
                    print(data.get_data().to_csv(index=False))
            else:
                if self._args.page is not None:
                    data.save_page_data(self._args.page, base_filename + '.csv')
                else:
                    data.save_data(base_filename + '.csv')
                    self._logger.info(f'csv output saved to {base_filename}.csv')

    def __save_text(self, text:str, filename: str):
        try:
            with open(filename, 'w') as f:
                f.write(text)
        except FileNotFoundError as e:
            raise CLIException("Output directory not found")

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
    _logger.info(' '.join(sys.argv))

    try:
        controller = CLIController()
        controller.run()
    except CLIException as e:
        print(e)
        _logger.error(e)
        exit(1)