"""Script to generate an embeddings database from pdf files."""

from dataclasses import dataclass
import argparse
import sys
import os
import glob

import yaml

from utils.controllers import CLI
from utils.exceptions import CLIException
from document_loaders.pdf import PyPDFMixedLoader, PyPDFLoader, OCRLoader
from document_splitters.hierarchical import TextTreeSplitter, DataTreeSplitter, TreeSplitter
from embeddings.storage import ChromaDBStorage

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

@dataclass
class ExecSettings:
    """Class to store settings for running the script."""
    cache_dir: str
    database_dir: str
    keep_cache: bool

@dataclass
class CollectionParams:
    """Class to store params for the creatinon of a collection."""
    embedder: str
    extraction_type: str
    inner_splitter: str
    loader: str

class CLIController(CLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME)

        self._args = self.__process_args()

    def run(self):
        """Run the script logic."""
        if self._args.settings_file == '':
            settings = ExecSettings(
                self._args.cache_dir,
                self._args.database_dir,
                self._args.keep_cache
            )
            collection_params = CollectionParams(
                self._args.embedder,
                self._args.extraction_type,
                self._args.inner_splitter,
                self._args.loader
            )

            if self._args.file != '':
                self.__process_file(self._args.file, self._args.collection, settings,
                                    collection_params)
            elif self._args.directory != '':
                self.__process_directory(self._args.directory, self._args.collection, settings,
                                     collection_params)
            else:
                raise CLIException("Input not specified")
        else:
            self.__process_yaml(self._args.settings_file)

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description=__doc__,
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>',
            formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument('-c', '--collection',
                            default='',
                            type=str,
                            help='Name of the collection to be created')
        parser.add_argument('--cache-dir',
                            default=DEFAULTS['cache_dir'],
                            type=str,
                            help=f'''
                                Directory to be used as cache.
                                Defaults to {DEFAULTS['cache_dir']}
                                ''')
        parser.add_argument('-d', '--directory',
                            default='',
                            type=str,
                            help='Directory to be processed in directory mode')
        parser.add_argument('--database-dir',
                            default=DEFAULTS['database_dir'],
                            type=str,
                            help=f'''
                                Directory to store the database.
                                Defaults to {DEFAULTS['database_dir']}
                                ''')
        parser.add_argument('-e', '--embedder',
                            default=DEFAULTS['embedder'],
                            type=str,
                            help=f'''Embeddings model to be used. Check SentenceTransformers
                                doc for all the options (
                                https://sbert.net/docs/sentence_transformer/pretrained_models.html
                                ). Defaults to {DEFAULTS['embedder']}
                                ''')
        parser.add_argument('--extraction-type',
                            default=DEFAULTS['extraction_type'],
                            choices=EXTRACTION_TYPES,
                            type=str,
                            help='Type of extraction to be performed. Defaults to text')
        parser.add_argument('-f', '--file',
                            default='',
                            type=str,
                            help='File to be processed in single file mode')
        parser.add_argument('--inner-splitter',
                            default=DEFAULTS['inner_splitter'],
                            choices=INNER_SPLITTERS,
                            help=f'''
                                Once sections are detected by the splitter, indicates how the
                                sections should be subdivided. Defaults to
                                {DEFAULTS['inner_splitter']}
                                ''')
        parser.add_argument('-k', '--keep-cache',
                            default=DEFAULTS['keep_cache'],
                            action='store_true',
                            help='''
                                Keep Tesseract cache after processing. Usefull when the same file
                                is going to be processed multiple times
                                ''')
        parser.add_argument('-l', '--loader',
                            default=DEFAULTS['loader'],
                            type=str,
                            choices=LOADERS.keys(),
                            help='Type of loader to use. Defaults to mixed')
        parser.add_argument('--settings-file',
                            default='',
                            type=str,
                            help='File with all the options to build a database')
        parser.add_argument('-v', '--version', action='version', version=VERSION)

        args = parser.parse_args()

        # If a settings file is used, all arguments are ignored
        if args.settings_file != '':
            return args

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
            raise CLIException(str(f"Incompatible extraction_type '{args.extraction_type}' "
                                   "with loader '{args.loader}'"))

        if args.collection == '':
            raise CLIException("Please specify a collection name")

        return args

    def __process_file(self, filename: str, collection:str, settings:ExecSettings,
                       params:CollectionParams):
        self._logger.info('Processing file %s', filename)

        basename = os.path.splitext(os.path.split(filename)[-1])[0]
        pdf_loader = self.__get_loader(filename, params.loader, settings.cache_dir,
                                       settings.keep_cache)
        if params.extraction_type == 'text':
            self._logger.info('Extracting text from file')

            text = pdf_loader.get_text()
            splitter = TextTreeSplitter(text, basename)
        elif params.extraction_type == 'data':
            self._logger.info('Extracting data from file')

            data = pdf_loader.get_document_data()
            splitter = DataTreeSplitter(data.get_data(remove_headers=True), basename)
        else:
            raise CLIException(f"Invalid extraction type '{params.extraction_type}'")

        self._logger.info('Obtaining file structure')
        splitter.analyze()
        sentences, metadatas = self.__extract_info(splitter)

        self._logger.info('Storing file info into Chromadb')
        storage = ChromaDBStorage(params.embedder, settings.database_dir)
        storage.save_info(collection, sentences, metadatas)

        self._logger.info('File %s processed', filename)

    def __process_directory(self, directory:str, collection:str, settings:ExecSettings,
                            params:CollectionParams):
        self._logger.info("Processing directory '%s'", directory)
        for file in glob.glob(os.path.join(directory, '*.pdf')):
            self.__process_file(file, collection, settings, params)

    def __process_yaml(self, yaml_file:str):
        yaml_settings = self.__load_settings(yaml_file)
        self.__validate_and_fill_settings(yaml_settings)

        settings = ExecSettings(
            yaml_settings['db']['settings']['cache_dir'],
            yaml_settings['db']['settings']['database_dir'],
            yaml_settings['db']['settings']['keep_cache'])

        if 'directory' in yaml_settings['db']:
            for collection, params in yaml_settings['db']['collections'].items():
                collection_params = CollectionParams(**params)
                self.__process_directory(yaml_settings['db']['directory'], collection, settings,
                                         collection_params)
        elif 'file' in yaml_settings['db']:
            for collection, params in yaml_settings['db']['collections'].items():
                collection_params = CollectionParams(**params)
                self.__process_file(yaml_settings['db']['file'], collection, settings,
                                    collection_params)
        else:
            raise CLIException("No file or directory to process was specified in settings file")

    def __get_loader(self, filename:str, loader:str, cache_dir:str, keep_cache:bool):
        self._logger.info("Using '%s' loader", loader)

        if loader == 'mixed':
            pdf_loader = PyPDFMixedLoader(cache_dir, keep_cache)
            pdf_loader.load(filename)
        elif loader == 'text':
            pdf_loader = PyPDFLoader(filename)
        elif loader == 'ocr':
            pdf_loader = OCRLoader(filename, cache_dir, keep_cache)
        else:
            raise CLIException("Invalid type of loader")

        return pdf_loader

    def __extract_info(self, splitter:TreeSplitter):
        sentences = []
        metadatas = []
        documents = splitter.extract_documents(self._args.inner_splitter)
        for doc in documents:
            sentences.append(doc.get_content())
            metadatas.append(doc.get_metadata())

        return sentences, metadatas

    def __load_settings(self, settings_file: str):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
        except yaml.scanner.ScannerError as e:
            self._logger.error(e)
            raise CLIException(f'{settings_file} is not a YAML file') from e

        return settings

    def __validate_and_fill_settings(self, settings):
        # Validate root node
        db = settings.get('db', None)
        if db is None:
            raise CLIException("'db' root node not found in settings file")

        # Validate directory or file
        if 'file' not in db and 'directory' not in db:
            raise CLIException("No file or directory to process was found in settings file")
        if 'file' in db and not os.path.exists(db['file']):
            raise CLIException(f"File '{db['file']}' not found")
        if 'directory' in db and not os.path.exists(db['directory']):
            raise CLIException(f"Directory '{db['directory']}' not found")

        # Validate settings node
        settings = db.get('settings', None)
        if settings is None:
            raise CLIException("'settings' node not found in settings file")

        # Validate cache directory
        if 'cache_dir' not in settings:
            settings['cache_dir'] = DEFAULTS['cache_dir']
        basedir = os.path.split(settings['cache_dir'])[0]
        if not os.path.exists(basedir):
            raise CLIException("Cache parent directory should exist")

        # Validate database directory
        if 'database_dir' not in settings:
            settings['database_dir'] = DEFAULTS['database_dir']
        basedir = os.path.split(settings['database_dir'])[0]
        if not os.path.exists(basedir):
            raise CLIException("Database parent directory should exist")

        # Validate keep cache flag
        if 'keep_cache' not in settings:
            settings['keep_cache'] = DEFAULTS['keep_cache']

        # Validating each collection node
        collections = db.get('collections', None)
        if collections is None:
            raise CLIException("No collections were specified")
        for _, params in collections.items():
            self.__validate_collection_params(params)

    def __validate_collection_params(self, params:dict):
        if 'embedder' not in params:
            params['embedder'] = DEFAULTS['embedder']

        if 'extraction_type' not in params:
            params['extraction_type'] = DEFAULTS['extraction_type']
        if params['extraction_type'] not in EXTRACTION_TYPES:
            raise CLIException(f"Invalid extraction_type '{params['extraction_type']}'")

        if 'inner_splitter' not in params:
            params['inner_splitter'] = DEFAULTS['inner_splitter']
        if params['inner_splitter'] not in INNER_SPLITTERS:
            raise CLIException(f"Invalid inner_splitter '{params['inner_splitter']}'")

        if 'loader' not in params:
            params['loader'] = DEFAULTS['loader']
        if params['loader'] not in LOADERS:
            raise CLIException(f"Invalid loader '{params['loader']}'")


if __name__ == "__main__":
    try:
        controller = CLIController()
        controller.run()
    except CLIException as e:
        print(e)
        controller.get_logger().error(e)
        sys.exit(1)
