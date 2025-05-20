import argparse
import os
import dotenv

from utils.logger import AppLogger
from embeddings.storage import ChromaDBStorage

dotenv.load_dotenv()

PROGRAM_NAME = 'GetRelevantCLI'
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

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Given a sentence, retrieves relevant documents from a vectorized documents database',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('sentence', type=str, help='Reference sentence to retrieve similar documents')

        parser.add_argument('-c', '--collection', default='', type=str, help='Name of the collection to search in the database')
        parser.add_argument('-d', '--database-dir', default='.db/', type=str, help='Database directory to be used')
        parser.add_argument('-e', '--embedder', default='all-MiniLM-L6-v2', type=str, help='Embeddings model to be used. Check SentenceTransformers doc for all the options (https://sbert.net/docs/sentence_transformer/pretrained_models.html)')
        parser.add_argument('-n', '--number-results', default=5, type=int, help='Number of relevant documents to retrieve')
        parser.add_argument('--version', action='store_true', help='Show version of this tool')

        args = parser.parse_args()

        if args.sentence == '':
            raise CLIException("Please specify a reference sentence")

        if args.collection == '':
            raise CLIException("Please specify a collection")

        if not os.path.exists(args.database_dir):
            raise CLIException(f"Database folder '{args.database_dir}' not found")

        return args

    def run(self):
        storage = ChromaDBStorage(self._args.embedder, self._args.database_dir)
        documents = storage.query_sentence(self._args.collection, self._args.sentence, self._args.number_results)

        for doc in documents:
            print("\n---------------------------------------")
            print(f"Sentence: {doc['content']}")
            print(f"Path: {doc['metadata']['path']}")
            print(f"Embeddings size: {doc['embeddings'].shape}")

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