"""Script to process a sentence and retrieve relevant documents from a vectorized
documents database.

This script requires a database previously created with get_embeddings.py script.
"""
import argparse
import os

from simplerag.llms.storage import ChromaDBStorage
from simplerag.llms.embedders import EmbedderParams
from .utils.controllers import CLI, run_cli
from .utils.exceptions import CLIException
from .utils.defaults import Defaults, DefaultEmbeddingParams

PROGRAM_NAME = 'GetRelevantCLI'
VERSION = '1.00.00'

class GetRelevantCLI(CLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

        self._args = None

    def run(self):
        """Run the script logic."""
        self._logger.debug('Loading database')
        embedder_params = EmbedderParams(embedding_context=self._args.embedding_context)
        storage = ChromaDBStorage(self._args.embedder,
                                  self._args.database_dir,
                                  embedder_params=embedder_params)

        self._logger.debug('Querying sentences')
        documents, _ = storage.query_sentence(
            self._args.collection,
            self._args.sentence,
            self._args.number_results)

        self._logger.debug('Showing results')
        for doc in documents:
            print("\n---------------------------------------")
            print(f"Sentence: {doc.get_content()}")
            print(f"Path: {doc.get_metadata()['path']}")
            print(f"Embeddings size: {doc.get_embeddings().shape}")

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
                            default=Defaults.database_dir,
                            type=str,
                            help=f'''
                                Database directory to be used.
                                Defaults to {Defaults.database_dir}
                                ''')
        self.parser.add_argument('-e', '--embedder',
                            default=Defaults.embedder,
                            type=str,
                            help=f'''Embeddings model to be used. Check SentenceTransformers doc
                                for all the options (
                                https://sbert.net/docs/sentence_transformer/pretrained_models.html
                                ). Defaults to {Defaults.embedder}''')
        self.parser.add_argument('--embedding-context',
                            default=DefaultEmbeddingParams.embedding_context,
                            type=int,
                            help='Context lenght for embeddings')
        self.parser.add_argument('-n', '--number-results',
                            default=Defaults.chat_num_related_docs,
                            type=int,
                            help=f'''
                                Number of relevant documents to retrieve.
                                Defaults to {Defaults.chat_num_related_docs}
                                ''')

        args = self.parser.parse_args()

        if args.sentence == '':
            raise CLIException("Please specify a reference sentence")

        if args.collection == '':
            raise CLIException("Please specify a collection")

        if not os.path.exists(args.database_dir):
            raise CLIException(f"Database folder '{args.database_dir}' not found")

        if args.embedding_context <= 0:
            raise CLIException("Embedding context must be greater than 0")

        self._args = args

def main():
    """Run the script."""
    run_cli(GetRelevantCLI)

if __name__ == "__main__":
    run_cli(GetRelevantCLI)
