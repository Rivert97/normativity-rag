"""Script to process a sentence and retrieve relevant documents from a vectorized
documents database.

This script requires a database previously created with get_embeddings.py script.
"""
import argparse
import os

from utils.controllers import CLI, run_cli
from utils.exceptions import CLIException
from llms.storage import ChromaDBStorage

PROGRAM_NAME = 'GetRelevantCLI'
VERSION = '1.00.00'

class CLIController(CLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

    def run(self):
        """Run the script logic."""
        self._logger.debug('Loading database')
        storage = ChromaDBStorage(self._args.embedder, self._args.database_dir)

        self._logger.debug('Querying sentences')
        documents = storage.query_sentence(
            self._args.collection,
            self._args.sentence,
            self._args.number_results)

        self._logger.debug('Showing results')
        for doc in documents:
            print("\n---------------------------------------")
            print(f"Sentence: {doc['content']}")
            print(f"Path: {doc['metadata']['path']}")
            print(f"Embeddings size: {doc['embeddings'].shape}")

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('sentence',
                            type=str,
                            help='Reference sentence to retrieve similar documents')

        self.parser.add_argument('-c', '--collection',
                            default='',
                            type=str,
                            help='Name of the collection to search in the database')
        self.parser.add_argument('-d', '--database-dir',
                            default='./db',
                            type=str,
                            help='Database directory to be used. Defaults to ./db')
        self.parser.add_argument('-e', '--embedder',
                            default='all-MiniLM-L6-v2',
                            type=str,
                            help='''Embeddings model to be used. Check SentenceTransformers doc
                                for all the options (
                                https://sbert.net/docs/sentence_transformer/pretrained_models.html
                                ). Defaults to all-MiniLM-L6-v2''')
        self.parser.add_argument('-n', '--number-results',
                            default=5,
                            type=int,
                            help='Number of relevant documents to retrieve. Defaults to 5')

        args = self.parser.parse_args()

        if args.sentence == '':
            raise CLIException("Please specify a reference sentence")

        if args.collection == '':
            raise CLIException("Please specify a collection")

        if not os.path.exists(args.database_dir):
            raise CLIException(f"Database folder '{args.database_dir}' not found")

        self._args = args

if __name__ == "__main__":
    run_cli(CLIController)
