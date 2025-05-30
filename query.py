"""Script to perform RAG with a custom database."""

import argparse
import os

from utils.controllers import CLI, run_cli
from utils.exceptions import CLIException
from llms.storage import ChromaDBStorage
from llms.rag import RAG
from llms.models import Qwen, LLama

DEFAULTS = {
    'embedder': 'all-MiniLM-L6-v2',
    'model': 'qwen',
}

MODELS = {
    'qwen': Qwen,
    'llama': LLama,
}

PROGRAM_NAME = 'RAG'
VERSION = '1.00.00'

class CLIController(CLI):
    """This class controls the execution of the program when used as CLI."""
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

        self._args = None

    def run(self):
        """Run the script logic."""
        model = MODELS[self._args.model]()

        if self._args.collection == '':
            rag = RAG(model=model)
            response = rag.query(self._args.query)
        else:
            storage = ChromaDBStorage(model=self._args.embedder, db_path=self._args.database_dir)
            rag = RAG(model=model, storage=storage)
            response = rag.query_with_documents(self._args.query, self._args.collection)

        print(response)

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('query',
                                 default='',
                                 type=str,
                                 help='Sentence query to be answered by the model')

        self.parser.add_argument('-c', '--collection',
                                 default='',
                                 type=str,
                                 help='Name of the collection to use. Must exist in the database')
        self.parser.add_argument('-d', '--database-dir',
                                default='',
                                type=str,
                                help='Directory where the database is stored')
        self.parser.add_argument('-e', '--embedder',
                                 default=DEFAULTS['embedder'],
                                 type=str,
                                 help=f'''
                                    Embeddings model to be used. Must match the database embedder.
                                    Defaults to {DEFAULTS['embedder']}
                                    ''')
        self.parser.add_argument('-m', '--model',
                                 default=DEFAULTS['model'],
                                 type=str,
                                 choices=MODELS.keys(),
                                 help=f'''
                                    Model to use as a conversational agent.
                                    Defaults to {DEFAULTS['model']}
                                    ''')

        args = self.parser.parse_args()

        if args.query == '':
            raise CLIException("Please specify an input query.")

        if args.database_dir != '' and not os.path.exists(args.database_dir):
            raise CLIException(f"Database directory '{args.database_dir}' not found")

        self._args = args

if __name__ == "__main__":
    run_cli(CLIController)
