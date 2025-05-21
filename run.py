import argparse
import dotenv
import sys
import os
import yaml
import glob

from utils.logger import AppLogger
from utils.exceptions import CLIException
from document_loaders.pdf import PyPDFMixedLoader, PyPDFLoader, OCRLoader
from document_splitters.hierarchical import TextTreeSplitter, DataTreeSplitter, TreeSplitter
from embeddings.storage import ChromaDBStorage

dotenv.load_dotenv()

PROGRAM_NAME = 'Runner'
VERSION = '1.00.00'

DEFAULTS = {
    'cache_dir': './.cache',
    'keep_cache': False,
    'loader': 'mixed',
    'extraction_type': 'text',
    'database_dir': './db',
    'embedder': 'all-MiniLM-L6-v2',
    'inner_splitter': 'paragraph',
}
LOADERS = {
    'mixed': PyPDFMixedLoader,
    'text': PyPDFLoader,
    'ocr': OCRLoader,
}
INNER_SPLITTERS = ['paragraph', 'section']
EXTRACTION_TYPES = ['text', 'data']

class CLIController():
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        self._logger = AppLogger.get_logger(PROGRAM_NAME)

        self._args = self.__process_args()

    def run(self):
        if self._args.file != '':
            self.__process_file(self._args.file)
        elif self._args.directory != '':
            self.__process_directory(self._args.directory)
        elif self._args.settings_file != '':
            self.__process_yaml(self._args.settings_file)
        else:
            raise CLIException("Input not specified")

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Generates an embeddings database from pdf files',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('-c', '--collection', default='', type=str, help='Name of the collection to be created')
        parser.add_argument('--cache-dir', default=DEFAULTS['cache_dir'], type=str, help=f'Directory to be used as cache. Defaults to {DEFAULTS['cache_dir']}')
        parser.add_argument('-d', '--directory', default='', type=str, help='Directory to be processed in directory mode')
        parser.add_argument('--database-dir', default=DEFAULTS['database_dir'], type=str, help=f'Directory to store the database. Defaults to {DEFAULTS['database_dir']}')
        parser.add_argument('-e', '--embedder', default=DEFAULTS['embedder'], type=str, help=f'Embeddings model to be used. Check SentenceTransformers doc for all the options (https://sbert.net/docs/sentence_transformer/pretrained_models.html). Defaults to {DEFAULTS['embedder']}')
        parser.add_argument('--extraction-type', default=DEFAULTS['extraction_type'], choices=EXTRACTION_TYPES, type=str, help='Type of extraction to be performed. Defaults to text')
        parser.add_argument('-f', '--file', default='', type=str, help='File to be processed in single file mode')
        parser.add_argument('--inner-splitter', default=DEFAULTS['inner_splitter'], choices=INNER_SPLITTERS, help=f'Once sections are detected by the splitter, indicates how the sections should be subdivided. Defaults to {DEFAULTS['inner_splitter']}')
        parser.add_argument('-k', '--keep-cache', default=DEFAULTS['keep_cache'], action='store_true', help='Keep Tesseract cache after processing. Usefull when the same file is going to be processed multiple times')
        parser.add_argument('-l', '--loader', default=DEFAULTS['loader'], type=str, choices=LOADERS.keys(), help='Type of loader to use. Defaults to mixed')
        parser.add_argument('--settings-file', default='', type=str, help='File with all the options to build a database')
        parser.add_argument('-v', '--version', action='version', version=VERSION)

        args = parser.parse_args()

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"Input file '{args.file}' not found")

        if args.directory != '' and not os.path.exists(args.directory):
            raise CLIException(f"Input directory '{args.directory}' not found")

        if args.settings_file != '' and not os.path.exists(args.settings_file):
            raise CLIException(f'Settings file "{args.settings_file}" not found')

        if args.file == '' and args.directory == '' and args.settings_file == '':
            raise CLIException("Please specify an input file, directory or settings file")

        args.cache_dir = args.cache_dir.rstrip('/')
        parent = os.path.split(args.cache_dir)[0]
        if parent != '' and not os.path.exists(parent):
            raise CLIException("Parent cache directory must exist")

        if args.extraction_type == 'data' and args.loader == 'text':
            raise CLIException(f"Incompatible extraction-type '{args.extraction_type}' with loader '{args.loader}'")

        if args.collection == '':
            raise CLIException("Please specify a collection name")

        return args

    def __process_file(self, filename: str):
        self._logger.info(f'Processing file {filename}')

        pdf_loader = self.__get_loader(filename)
        if self._args.extraction_type == 'text':
            self._logger.info('Extracting text from file')

            text = pdf_loader.get_text()
            splitter = TextTreeSplitter(text, filename)
        elif self._args.extraction_type == 'data':
            self._logger.info('Extracting data from file')

            data = pdf_loader.get_document_data()
            splitter = DataTreeSplitter(data.get_data(remove_headers=True), filename)
        else:
            raise CLIException(f"Invalid extraction type '{self._args.extraction_type}'")

        self._logger.info('Obtaining file structure')
        splitter.analyze()
        sentences, metadatas = self.__extract_info(splitter)

        self._logger.info('Storing file info into Chromadb')
        storage = ChromaDBStorage(self._args.embedder, self._args.database_dir)
        storage.save_info(self._args.collection, sentences, metadatas)

        self._logger.info(f'File {filename} processed')

    def __process_directory(self, directory:str):
        self._logger.info(f'Processing directory {directory}')
        for file in glob.glob(os.path.join(directory, '*.pdf')):
            self.__process_file(file)

    def __process_yaml(self, yaml_file:str):
        pass

    def __get_loader(self, filename:str):
        self._logger.info(f'Using "{self._args.loader}" loader')

        if self._args.loader == 'mixed':
            pdf_loader = PyPDFMixedLoader(self._args.cache_dir, self._args.keep_cache)
            pdf_loader.load(filename)
        elif self._args.loader == 'text':
            pdf_loader = PyPDFLoader(filename)
        elif self._args.loader == 'ocr':
            pdf_loader = OCRLoader(filename, self._args.cache_dir, self._args.keep_cache)
        else:
            raise CLIException("Invalid type of loader")

        return pdf_loader

    def __get_documents_from_text(self, text:str, name:str):
        splitter = TextTreeSplitter(text, name)
        splitter.analyze()
        sentences, metadatas = self.__extract_info(splitter)

    def __get_documents_from_data(self):
        pass

    def __extract_info(self, splitter:TreeSplitter):
        sentences = []
        metadatas = []
        documents = splitter.extract_documents(self._args.inner_splitter)
        for doc in documents:
            sentences.append(doc.get_content())
            metadatas.append(doc.get_metadata())

        return sentences, metadatas

    def __load_settings(self):
        try:
            with open(self._args.settings_file, 'r') as f:
                settings = yaml.safe_load(f)
        except yaml.scanner.ScannerError as e:
            self._logger.error(e)
            raise CLIException(f'{self._args.settings_file} is not a YAML file')

        return settings

    def __validate_settings(self):
        # Validate root node
        db = self._settings.get('db', None)
        if db is None:
            raise CLIException('"db" root node not found in settings file')

        # Validate settings node
        settings = db.get('settings', None)
        if settings is None:
            raise CLIException('"settings" node not found in settings file')

        # Validate database directory
        if 'db-dir' not in settings:
            raise CLIException('"db-dir" not found in settings node')
        basedir = os.path.split(settings['db-dir'])[0]
        if not os.path.exists(basedir):
            raise CLIException('Database parent directory should exist')

        # Validate source directory
        if 'directory' not in settings:
            raise CLIException('"directory" not found in settings node')
        if not os.path.exists(settings['directory']):
            raise CLIException(f'Direcotry {settings['directory']} not found')

        # Validating each collection node
        collections = db.get('collections', None)
        if collections is None:
            raise CLIException('No collections specified')
        for collection, params in collections.items():
            loader = params.get('loader', None)
            if loader is None:
                raise CLIException(f'No loader specified in collection "{collection}"')
            if loader not in LOADERS:
                raise CLIException(f'Invalid loader "{loader}" in collection "{collection}"')

            inner_splitter = params.get('inner-splitter', None)
            if inner_splitter is None:
                raise CLIException(f'No inner-splitter specified in collection "{collection}"')
            if inner_splitter not in INNER_SPLITTERS:
                raise CLIException(f'Invalid inner-splitter "{inner_splitter}" in colleciton "{collection}"')

            model = params.get('model', None)
            if model is None:
                raise CLIException(f'No model specified in collection "{collection}"')

    def __create_collection(self, name:str, parameters:dict[str, str], settings:dict[str, str]):
        for file in glob.glob(os.path.join(settings['directory'], '*.pdf')):
            pdf_loader = LOADERS[parameters['loader']]()
            pdf_loader.load(file)


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