"""Scritp to get embeddings from a text file or a csv data file from a PDF.

This script is intended to be used by it's own when processing simple text files
or alogside extract_info.py when using PDF files to generate a txt or csv file.
"""
import argparse
import os
import glob
import sys

from utils.controllers import CLI
from utils.exceptions import CLIException
from document_loaders.representations import PdfDocumentData
from document_splitters.hierarchical import TreeSplitter, DataTreeSplitter, TextTreeSplitter
from embeddings.embedders import STEmbedder
from embeddings.storage import CSVStorage, ChromaDBStorage

PROGRAM_NAME = 'EmbeddingsCLI'
VERSION = '1.00.00'

class CLIController(CLI):
    """Class to control the execution of the program when usied as CLI."""

    def __init__(self):
        """Initialize the controller.

        Create the logger for the script and process the CLI arguments.
        """
        super().__init__(PROGRAM_NAME)

        self.print_to_console = True
        self.storage = None
        self.embedder = None

        self._args = self.__process_args()

    def run(self):
        """Run the script logic."""
        if self._args.file != '':
            self.__process_file(self._args.file, self._args.output, self._args.type)
        elif self._args.directory != '':
            self.__process_directory(self._args.directory)
        else:
            raise CLIException("Input not specified")

    def __process_directory(self, directory: str):
        for file in glob.glob(f'{directory}/*.{self._args.type}'):
            basename = ''.join(os.path.basename(file).split('.')[:-1])
            if self._args.page is None:
                out_name = os.path.join(self._args.output, f"{basename}")
            else:
                out_name = os.path.join(self._args.output, f"{basename}_{self._args.page}")

            self.__process_file(file, out_name, self._args.type)

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description=__doc__,
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>',
            formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument('-a', '--action',
                            default='embeddings',
                            choices=['embeddings', 'structure', 'tree'],
                            type=str,
                            help='''
                                Action to perform.
                                embeddings: Split the file and get the embeddings.
                                structure: Show file structure in console.
                                tree: Show an image of the tree of titles of the file.
                                Defaults to embeddings
                                ''')
        parser.add_argument('-c', '--collection',
                            default='',
                            type=str,
                            help='''
                                When using embeddings action and storage is not csv,
                                name of the collection where the embeddings should be stored
                                ''')
        parser.add_argument('-d', '--directory',
                            default='',
                            type=str,
                            help='Directory to be processed in directory mode')
        parser.add_argument('--database-dir',
                            default='./db',
                            type=str,
                            help='Directory to store the database. Defaults to ./db')
        parser.add_argument('-e', '--embedder',
                            default='all-MiniLM-L6-v2',
                            type=str,
                            help='''
                                Embeddings model to be used. Check SentenceTransformers doc for
                                all the options (
                                https://sbert.net/docs/sentence_transformer/pretrained_models.html
                                ). Defaults to all-MiniLM-L6-v2
                                ''')
        parser.add_argument('-f', '--file',
                            default='',
                            type=str,
                            help='Path to file containing the data or text of the document')
        parser.add_argument('-o', '--output', default='', help='Name of the file to be saved')
        parser.add_argument('-p', '--page', type=int, help='Number of page to be processed')
        parser.add_argument('-s', '--storage',
                            default='csv',
                            type=str,
                            choices=['csv', 'chromadb'],
                            help='Type of storage to be used for embeddings. Defaults to csv')
        parser.add_argument('--inner-splitter',
                            default='paragraph',
                            choices=['paragraph', 'section'],
                            help='''
                                Once sections are detected by the splitter, indicates how the
                                sections should be subdivided. Defaults to paragraph
                                ''')
        parser.add_argument('-t', '--type',
                            default='csv',
                            choices=['csv', 'txt'],
                            type=str,
                            help='Type of input. Defaults to csv')
        parser.add_argument('-v', '--version', action='version', version=VERSION)

        args = parser.parse_args()

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"File '{args.file} not found")

        if args.directory != '' and not os.path.exists(args.directory):
            raise CLIException(f"Input directory '{args.directory}' not found")

        basedir = os.path.dirname(args.output)
        if args.output != '' and basedir != '' and not os.path.exists(basedir):
            raise CLIException(f"Output path '{args.output}' does not exist")

        if args.file == '' and args.directory == '':
            raise CLIException("Please specify an input file or directory")

        if args.directory != '':
            if args.output == '':
                args.output = './'
            if not os.path.isdir(args.output):
                raise CLIException("Destination directory does not exist")

        if args.output != '' or (args.action == 'embeddings' and args.storage != 'csv'):
            self.print_to_console = False

        if args.storage == 'csv':
            self.storage = CSVStorage()
        elif args.storage == 'chromadb':
            self.storage = ChromaDBStorage(args.embedder, args.database_dir)
        else:
            raise CLIException(f"Invalid storage '{args.storage}'")

        if args.action == 'embeddings' and args.storage != 'csv' and args.collection == '':
            raise CLIException("Please specify a name for the collection")

        return args

    def __process_file(self, filename: str, output: str, input_type: str):
        self._logger.info('Processing file %s', filename)

        if input_type == 'csv':
            splitter = self.__load_and_split_doc(filename)
        elif input_type == 'txt':
            splitter = self.__load_and_split_txt(filename)
        else:
            raise CLIException(f"Invalid type of file: '{input_type}'")

        if self._args.action == 'embeddings':
            self.__action_embeddings(splitter, output)
        elif self._args.action == 'structure':
            self.__action_structure(splitter, output)
        elif self._args.action == 'tree':
            self.__action_tree(splitter, output)
        else:
            raise CLIException(f"Invalid action '{self._args.action}'")

    def __action_embeddings(self, splitter:TreeSplitter, output: str):
        self._logger.info('Loading embeddings')

        sentences, metadatas = self.__extract_info(splitter)

        if self._args.storage == 'csv':
            try:
                self.embedder = STEmbedder(self._args.embedder)
            except OSError as e:
                raise CLIException(f"Invalid embedder '{self._args.embedder}'") from e

            embeddings = self.embedder.get_embeddings(sentences)
        else:
            embeddings = None

        if self.print_to_console:
            print('sentences,metadatas,embeddings')
            for sent, meta, emb in zip(sentences, metadatas, embeddings):
                print(f'{sent},{meta},{emb}')
        else:
            if self._args.storage == 'csv':
                name = os.path.splitext(output)[0] + '-embeddings.csv'
            else:
                name = self._args.collection

            self.storage.save_info(name, sentences, metadatas, embeddings)
            self._logger.info("Embeddings saved to '%s'", name)

    def __action_structure(self, splitter:TreeSplitter, output:str):
        self._logger.info('Loading file structure')

        if self.print_to_console:
            splitter.show_file_structure()
        else:
            base_filename = os.path.splitext(output)[0]
            structure = splitter.get_file_structure()
            self.__save_txt_file(base_filename + '-structure.txt', structure)
            self._logger.info('File structure save to %s-structure.txt', base_filename)

    def __action_tree(self, splitter:TreeSplitter, output:str):
        self._logger.info('Loading file tree')

        if self.print_to_console:
            splitter.show_tree()
        else:
            base_filename = os.path.splitext(output)[0]
            splitter.save_tree(base_filename + '-tree.png')
            self._logger.info('File tree saved to %s-tree.png', base_filename)

    def __load_and_split_doc(self, filename:str) -> DataTreeSplitter:
        self._logger.info('Loading document data')

        document_data = PdfDocumentData()
        document_data.load_data(filename)
        basename = os.path.splitext(os.path.split(filename)[-1])[0]
        if self._args.page is not None:
            splitter = DataTreeSplitter(
                    document_data.get_page_data(self._args.page, remove_headers=True),
                    basename)
        else:
            splitter = DataTreeSplitter(document_data.get_data(remove_headers=True), basename)

        splitter.analyze()

        return splitter

    def __load_and_split_txt(self, filename:str) -> TextTreeSplitter:
        self._logger.info('Loading text data')

        basename = os.path.splitext(os.path.split(filename)[-1])[0]
        with open(filename, 'r', encoding='utf-8') as f:
            file_content = f.read()

        splitter = TextTreeSplitter(file_content, basename)
        splitter.analyze()

        return splitter

    def __save_txt_file(self, filename:str, content:str):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

    def __extract_info(self, splitter:TreeSplitter):
        sentences = []
        metadatas = []
        documents = splitter.extract_documents(self._args.inner_splitter)
        for doc in documents:
            sentences.append(doc.get_content())
            metadatas.append(doc.get_metadata())

        return sentences, metadatas

if __name__ == "__main__":
    try:
        controller = CLIController()
        controller.run()
    except CLIException as e:
        print(e)
        controller.get_logger().error(e)
        sys.exit(1)
