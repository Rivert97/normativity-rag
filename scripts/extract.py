"""Script to generate an embeddings database from pdf files."""

from dataclasses import dataclass
import argparse
import os
import glob

from simplerag.document_loaders.pdf import PDFPlumberLoader
from simplerag.document_splitters.hierarchical import TreeSplitter
from simplerag.document_splitters.hierarchical import DataTreeSplitter
from simplerag.document_splitters.hierarchical import TextTreeSplitter
from simplerag.document_splitters.hierarchical import DataSplitterOptions
from simplerag.llms.storage import ChromaDBStorage
from .utils.controllers import CLI, run_cli
from .utils.exceptions import CLIException
from .utils.defaults import Defaults, DEFAULT_PARSE_PARAMS

PROGRAM_NAME = 'extract'
VERSION = '1.00.00'

INNER_SPLITTERS = ['paragraph', 'section']
EXTRACTION_TYPES = ['text', 'data']

@dataclass
class ExecSettings:
    """Class to store settings for running the script."""
    database_dir: str
    file_settings: dict[str,dict[str,str]]

@dataclass
class CollectionParams:
    """Class to store params for the creatinon of a collection."""
    embedder: str
    extraction_type: str
    inner_splitter: str
    raw: bool = False
    max_chars: int = Defaults.max_chars

class ExtractorCLI(CLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

        self._args = None

    def run(self):
        """Run the script logic."""
        if self._args.settings_file == '':
            settings = ExecSettings(
                self._args.database_dir,
                {
                    '*': {
                        'parse_params_file': self._args.parse_params_file,
                    }
                },
            )
            collection_params = CollectionParams(
                self._args.embedder,
                self._args.extraction_type,
                self._args.inner_splitter,
                self._args.raw,
                self._args.max_chars,
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

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('-c', '--collection',
                            default='',
                            type=str,
                            help='Name of the collection to be created')
        self.parser.add_argument('-d', '--directory',
                            default='',
                            type=str,
                            help='Directory to be processed in directory mode')
        self.parser.add_argument('--database-dir',
                            default=Defaults.database_dir,
                            type=str,
                            help=f'''
                                Directory to store the database.
                                Defaults to {Defaults.database_dir}
                                ''')
        self.parser.add_argument('-e', '--embedder',
                            default=Defaults.embedder,
                            type=str,
                            help=f'''Embeddings model to be used. Check SentenceTransformers
                                doc for all the options (
                                https://sbert.net/docs/sentence_transformer/pretrained_models.html
                                ). Defaults to {Defaults.embedder}
                                ''')
        self.parser.add_argument('--extraction-type',
                            default=Defaults.extraction_type,
                            choices=EXTRACTION_TYPES,
                            type=str,
                            help=f'''Type of extraction to be performed.
                                Defaults to {Defaults.extraction_type}''')
        self.parser.add_argument('-f', '--file',
                            default='',
                            type=str,
                            help='File to be processed in single file mode')
        self.parser.add_argument('--inner-splitter',
                            default=Defaults.inner_splitter,
                            choices=INNER_SPLITTERS,
                            help=f'''
                                Once sections are detected by the splitter, indicates how the
                                sections should be subdivided. Defaults to
                                {Defaults.inner_splitter}
                                ''')
        self.parser.add_argument('--max-chars',
                            default=Defaults.max_chars,
                            type=int,
                            help=f'''
                                Maximum number of characters per chunk. It will find nearest dot.
                                Defaults to {Defaults.max_chars}.
                                ''')
        self.parser.add_argument('--parse-params-file',
                            default='',
                            type=str,
                            help='''
                                YAML file with custom parse parameters to be used
                                during extraction
                                ''')
        self.parser.add_argument('--raw',
                                 default=False,
                                 action='store_true',
                                 help='''
                                     Use this option to use text as returned by the library.
                                 ''')
        self.parser.add_argument('--settings-file',
                            default='',
                            type=str,
                            help='''
                                File with all the options to build a database. Use this option to
                                store all options to process files when it will be repeated.
                                ''')

        args = self.parser.parse_args()

        # If a settings file is used, all arguments are ignored
        if args.settings_file != '':
            self._args = args
            return

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"Input file '{args.file}' not found")

        if args.directory != '' and not os.path.exists(args.directory):
            raise CLIException(f"Input directory '{args.directory}' not found")

        if args.settings_file != '' and not os.path.exists(args.settings_file):
            raise CLIException(f'Settings file "{args.settings_file}" not found')

        if args.file == '' and args.directory == '' and args.settings_file == '':
            raise CLIException("Please specify an input file, directory or settings file")

        if args.collection == '':
            raise CLIException("Please specify a collection name")

        if args.parse_params_file != '' and not os.path.exists(args.parse_params_file):
            raise CLIException("Parse parameters file does not exist")

        self._args = args

    def __process_file(self, filename: str, collection:str, settings:ExecSettings,
                       params:CollectionParams):
        self._logger.info('Processing file %s', filename)

        file_settings = self.__get_file_settings(filename, settings)
        file_parse_params = self.load_yaml(file_settings.get('parse_params_file', ''))
        if not file_parse_params:
            file_parse_params = DEFAULT_PARSE_PARAMS

        basename = os.path.splitext(os.path.split(filename)[-1])[0]
        pdf_loader = PDFPlumberLoader(filename, params.raw)
        if params.extraction_type == 'text':
            self._logger.info('Extracting text from file')
            text = pdf_loader.get_text(boundaries=file_parse_params.get('pdf_margins'))
            splitter = TextTreeSplitter(text, basename, params.max_chars)
        elif params.extraction_type == 'data':
            self._logger.info('Extracting data from file')

            data = pdf_loader.get_document_data()
            splitter = DataTreeSplitter(
                data.get_data(remove_headers=True, boundaries=file_parse_params.get('pdf_margins')),
                basename,
                DataSplitterOptions(max_characters=params.max_chars)
            )
        else:
            raise CLIException(f"Invalid extraction type '{params.extraction_type}'")

        self._logger.info('Obtaining file structure')
        splitter.analyze()
        sentences, metadatas = self.__extract_info(splitter, params)

        self._logger.info('Storing file info into Chromadb')
        storage = ChromaDBStorage(params.embedder, settings.database_dir)
        storage.save_info(
            collection,
            {
                'sentences': sentences,
                'metadatas': metadatas,
            },
            id_prefix=f'{filename}_')

        self._logger.info('File %s processed', filename)

    def __process_directory(self, directory:str, collection:str, settings:ExecSettings,
                            params:CollectionParams):
        self._logger.info("Processing directory '%s'", directory)
        for file in glob.glob(os.path.join(directory, '*.pdf')):
            self.__process_file(file, collection, settings, params)

    def __process_yaml(self, yaml_file:str):
        yaml_settings = self.load_yaml(yaml_file)
        self.__validate_and_fill_settings(yaml_settings)

        settings = ExecSettings(
            yaml_settings['db']['settings']['database_dir'],
            yaml_settings['db'].get('file_settings', {}))

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

    def __extract_info(self, splitter:TreeSplitter, params:CollectionParams):
        sentences = []
        metadatas = []
        documents = splitter.extract_documents(params.inner_splitter)
        for doc in documents:
            sentences.append(doc['content'])
            metadatas.append(doc['metadata'])

        return sentences, metadatas

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

        # Validate database directory
        if 'database_dir' not in settings:
            settings['database_dir'] = Defaults.database_dir
        basedir = os.path.split(settings['database_dir'])[0]
        if not os.path.exists(basedir):
            raise CLIException("Database parent directory should exist")

        # Validating each collection node
        collections = db.get('collections', None)
        if collections is None:
            raise CLIException("No collections were specified")
        for _, params in collections.items():
            self.__validate_collection_params(params)

    def __validate_collection_params(self, params:dict):
        if 'embedder' not in params:
            params['embedder'] = Defaults.embedder

        if 'extraction_type' not in params:
            params['extraction_type'] = Defaults.extraction_type
        if params['extraction_type'] not in EXTRACTION_TYPES:
            raise CLIException(f"Invalid extraction_type '{params['extraction_type']}'")

        if 'inner_splitter' not in params:
            params['inner_splitter'] = Defaults.inner_splitter
        if params['inner_splitter'] not in INNER_SPLITTERS:
            raise CLIException(f"Invalid inner_splitter '{params['inner_splitter']}'")

    def __get_file_settings(self, filename:str, settings:ExecSettings) -> dict[str,str]:
        default_settings = settings.file_settings.get('*', {})
        settings = settings.file_settings.get(os.path.split(filename)[-1], default_settings)

        return settings

def main():
    """Run the script."""
    run_cli(ExtractorCLI)

if __name__ == "__main__":
    run_cli(ExtractorCLI)
