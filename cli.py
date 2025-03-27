import argparse
import os
import glob

from Loaders import PdfMixedLoader

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
        self.args = self.__process_args()

    def run(self):
        if self.args.file != '':
            self.__process_file()
        elif self.args.directory != '':
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
        parser.add_argument('-o', '--output-dir', default='./', type=str, help='Directory to store the output text files. Defaults to ./')
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
        
        if args.output_dir != './' and not os.path.exists(args.output_dir):
            raise CLIException("Destination folder does not exist")
        
        return args
    
    def __process_file(self):
        basename = ''.join(os.path.basename(self.args.file).split('.')[:-1])

        pdf_loader = PdfMixedLoader(self.args.file, self.args.cache_dir)
        if self.args.page != None:
            text = pdf_loader.get_page_text(self.args.page)
            out_name = f"{self.args.output_dir}/{basename}_{self.args.page}.txt"
        else:
            text = pdf_loader.get_text()
            out_name = f"{self.args.output_dir}/{basename}.txt"

        with open(out_name, 'w') as f:
            f.write(text)
    
    def __process_directory(self):
        for file in glob.glob(f'{self.args.directory}/*.pdf'):
            basename = ''.join(os.path.basename(file).split('.')[:-1])

            pdf_loader = PdfMixedLoader(file)
            if self.args.page != None:
                text = pdf_loader.get_page_text(self.args.page)
                out_name = f"{self.args.output_dir}/{basename}_{self.args.page}.txt"
            else:
                text = pdf_loader.get_text()
                out_name = f"{self.args.output_dir}/{basename}.txt"

            with open(out_name, 'w') as f:
                f.write(text)

if __name__ == "__main__":
    try:
        controller = CLIController()
        controller.run()
    except CLIException as e:
        print(e)
        exit(1)