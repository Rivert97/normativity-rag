import argparse
import os
import dotenv
import glob

from utils.logger import AppLogger
from document_loaders.representations import PdfDocumentData
from document_splitters.hierarchical import TreeSplitter, DataTreeSplitter, TextTreeSplitter
from embeddings.embedders import STEmbedder
from embeddings.storage import CSVStorage, ChromaDBStorage

dotenv.load_dotenv()

PROGRAM_NAME = 'EmbeddingsCLI'
VERSION = '1.00.00'

STORAGES = {
    'csv': CSVStorage,
    'chromadb': ChromaDBStorage,
}

class CLIException(Exception):
    def __init__(self, message):
        super().__init__(f"{PROGRAM_NAME} ERROR: {message}")

class CLIController():
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        self._logger = AppLogger.get_logger('CLIController')

        self.print_to_console = True
        self.embedder = None
        self.storage = None

        self._args = self.__process_args()

    def run(self):
        if self._args.file != '':
            self.__process_file(self._args.file, self._args.output, self._args.type)
        elif self._args.directory != '':
            self.__process_directory(self._args.directory)
        else:
            raise CLIException("Input not specified")

    def __process_directory(self, directory: str):
        for file in glob.glob(f'{directory}/*.{self._args.type}'):
            basename = ''.join(os.path.basename(file).split('.')[:-1])
            if self._args.page != None:
                out_name = os.path.join(self._args.output, f"{basename}_{self._args.page}")
            else:
                out_name = os.path.join(self._args.output, f"{basename}")

            self.__process_file(file, out_name, self._args.type)

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Creates a vectorized database of data extracted from PDF files',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('-a', '--action', default='embeddings', choices=['embeddings', 'structure', 'tree'], type=str, help='Action to perform: "embeddings" to split the file and get the embeddings, "structure" to show file structure in console. "tree" to show an image of the tree of titles of the file.')
        parser.add_argument('-c', '--collection', default='', type=str, help='In embeddings mode and storage is not csv, name of the collection where the embeddings should be stored')
        parser.add_argument('-d', '--directory', default='', type=str, help='Directory to be processed in directory mode')
        parser.add_argument('-e', '--embedder', default='all-MiniLM-L6-v2', type=str, help='Embeddings model to be used. Check SentenceTransformers doc for all the options (https://sbert.net/docs/sentence_transformer/pretrained_models.html)')
        parser.add_argument('-f', '--file', default='', type=str, help='Path to file containing the data or text of the document')
        parser.add_argument('-o', '--output', default='', help='Name of the file to be saved')
        parser.add_argument('-p', '--page', type=int, help='Number of page to be processed')
        parser.add_argument('-s', '--storage', default='csv', type=str, choices=['csv', 'chromadb'], help='Type of storage to be used for embeddings')
        parser.add_argument('--inner-splitter', default='paragraph', choices=['paragraph', 'section'], help='Once sections are detected by the splitter, indicates how the sections should be subdivided')
        parser.add_argument('-t', '--type', default='csv', choices=['csv', 'txt'], type=str, help='Type of input')
        parser.add_argument('--version', action='store_true', help='Show version of this tool')

        args = parser.parse_args()

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"File '{args.file} not found")

        if args.directory != '' and not os.path.exists(args.directory):
            raise CLIException(f"Input directory '{args.directory}' not found")

        if args.output != '':
            basedir = os.path.dirname(args.output)
            if basedir != '' and not os.path.exists(basedir):
                raise CLIException(f"Output path '{args.output}' does not exist")

        if args.file == '' and args.directory == '':
            raise CLIException(f"Please specify an input file or directory")

        if args.directory != '':
            if args.output == '':
                args.output = './'
            if not os.path.isdir(args.output):
                raise CLIException("Destination directory does not exist")

        if args.output != '' or (args.action == 'embeddings' and args.storage != 'csv'):
            self.print_to_console = False

        try:
            self.embedder = STEmbedder(args.embedder)
        except OSError as e:
            raise CLIException(f"Invalid embedder '{args.embedder}'")

        if args.storage in STORAGES:
            self.storage = STORAGES[args.storage]
        else:
            raise CLIException(f"Invalid storage '{args.storage}'")

        if args.action == 'embeddings' and args.storage != 'csv' and args.collection == '':
            raise CLIException("Please specify a name for the collection")

        return args

    def __process_file(self, filename: str, output: str, type: str):
        if type == 'csv':
            splitter = self.__load_and_split_doc(filename)
        elif type == 'txt':
            splitter = self.__load_and_split_txt(filename)
        else:
            raise CLIException(f"Invalid type of file: '{type}'")

        if self._args.action == 'embeddings':
            self.__action_embeddings(splitter, output)
        elif self._args.action == 'structure':
            self.__action_structure(splitter, output)
        elif self._args.action == 'tree':
            self.__action_tree(splitter, output)
        else:
            raise CLIException(f"Invalid action '{self._args.action}'")

    def __action_embeddings(self, splitter:TreeSplitter, output: str):
        sentences, metadatas = self.__extract_info(splitter)

        if self._args.storage == 'csv':
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

            storage = self.storage()
            storage.save_info(name, sentences, metadatas, embeddings)

    def __action_structure(self, splitter:TreeSplitter, output:str):
        if self.print_to_console:
            splitter.show_file_structure()
        else:
            base_filename = os.path.splitext(output)[0]
            structure = splitter.get_file_structure()
            self.__save_txt_file(base_filename + '-structure.txt', structure)

    def __action_tree(self, splitter:TreeSplitter, output:str):
        if self.print_to_console:
            splitter.show_tree()
        else:
            base_filename = os.path.splitext(output)[0]
            splitter.save_tree(base_filename + '-tree.png')

    def __load_and_split_doc(self, filename:str) -> DataTreeSplitter:
        document_data = PdfDocumentData()
        document_data.load_data(filename)
        if self._args.page != None:
            splitter = DataTreeSplitter(document_data.get_page_data(self._args.page, remove_headers=True), filename)
        else:
            splitter = DataTreeSplitter(document_data.get_data(remove_headers=True), filename)

        splitter.analyze()

        return splitter

    def __load_and_split_txt(self, filename:str) -> TextTreeSplitter:
        with open(filename, 'r') as f:
            file_content = f.read()

        splitter = TextTreeSplitter(file_content, filename)
        splitter.analyze()

        return splitter

    def __save_txt_file(self, filename:str, content:str):
        with open(filename, 'w') as f:
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