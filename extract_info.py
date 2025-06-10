"""Script to load a PDF file and converts it to plain text or csv data with the
position information of each word.

When used to extract csv data, this script is intended to be used alogside
get_embeddings.py to obtain further information of the file.
"""
import argparse
import os
import glob

from document_loaders.pdf import PyPDFMixedLoader, PyPDFLoader, OCRLoader
from utils.controllers import CLI, run_cli
from utils.exceptions import CLIException

PROGRAM_NAME = 'ExtractorCLI'
VERSION = '1.00.00'

class CLIController(CLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

        self.print_to_console = True
        self._args = None

    def run(self):
        """Run the script logic."""
        if self._args.file != '':
            self.__process_file(self._args.file, self._args.output)
        elif self._args.directory != '':
            self.__process_directory()
        else:
            raise CLIException("Input not specified")

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('--cache-dir',
                            default='./.cache',
                            type=str,
                            help='Directory to be used as cache. Defaults to ./.cache')
        self.parser.add_argument('-d', '--directory',
                            default='',
                            type=str,
                            help='Directory to be processed in directory mode')
        self.parser.add_argument('-f', '--file',
                            default='',
                            type=str,
                            help='File to be processed in single file mode')
        self.parser.add_argument('-k', '--keep-cache',
                            default=False,
                            action='store_true',
                            help='''Keep Tesseract cache after processing. Usefull when the
                                same file is going to be processed multiple times''')
        self.parser.add_argument('-l', '--loader',
                            default='mixed',
                            type=str,
                            choices=['mixed', 'text', 'ocr'],
                            help='Type of loader to use. Defaults to mixed')
        self.parser.add_argument('-o', '--output',
                            default='',
                            type=str,
                            help='''File or directory to store the output text file(s).
                                When -d is used, this defaults to ./''')
        self.parser.add_argument('-p', '--page',
                            type=int,
                            help='Number of page to be processed')
        self.parser.add_argument('-t', '--type',
                            default='txt',
                            choices=['txt', 'csv'],
                            nargs='+',
                            type=str,
                            help='Type(s) of output(s). Defaults to txt')
        self.parser.add_argument('-P', '--Parallel',
                            default=False,
                            action="store_true",
                            help='Uses the total number of cores - 2, for concurrent processing')

        args = self.parser.parse_args()

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

        self._args = args

    def __process_file(self, filename: str, output: str = None):
        self._logger.info('Processing file %s', filename)

        pdf_loader = self.__get_loader(filename)
        self.__make_output(pdf_loader, output)

    def __process_directory(self):
        for file in glob.glob(f'{self._args.directory}/*.pdf'):
            basename = ''.join(os.path.basename(file).split('.')[:-1])
            if self._args.page is not None:
                out_name = f"{self._args.output}/{basename}_{self._args.page}"
            else:
                out_name = f"{self._args.output}/{basename}"

            self.__process_file(file, out_name)

    def __get_loader(self, filename:str):
        self._logger.info("Using '%s' loader", self._args.loader)

        if self._args.loader == 'mixed':
            loader = PyPDFMixedLoader(self._args.cache_dir, self._args.keep_cache)
            if self._args.page is not None:
                self._logger.info('Processing page %s', self._args.page)

                loader.load_page(filename, self._args.page)
            else:
                loader.load(filename, parallel=self._args.Parallel)
        elif self._args.loader == 'text':
            loader = PyPDFLoader(filename)
        elif self._args.loader == 'ocr':
            loader = OCRLoader(filename, self._args.cache_dir, self._args.keep_cache)
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
                self._logger.info('Text output saved to %s.txt', base_filename)

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
                    self._logger.info('csv output saved to %s.csv', base_filename)

    def __save_text(self, text:str, filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
        except FileNotFoundError as e:
            raise CLIException("Output directory not found") from e

if __name__ == "__main__":
    run_cli(CLIController)
