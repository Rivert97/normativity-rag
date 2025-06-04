"""Script to perform RAG with a custom database."""

import argparse
import os

from utils.controllers import CLI, run_cli
from utils.exceptions import CLIException
from llms.storage import ChromaDBStorage
from llms.rag import RAG
from llms.models import GemmaBuilder, LlamaBuilder, QwenBuilder, MistralBuilder
from llms.data import Document

DEFAULTS = {
    'embedder': 'all-MiniLM-L6-v2',
    'model': 'gemma',
}

MODEL_BUILDERS = {
    'gemma': GemmaBuilder,
    'llama': LlamaBuilder,
    'qwen': QwenBuilder,
    'mistral': MistralBuilder,
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
        self._logger.info("Loading Model '%s (%s)'", self._args.model, self._args.variant)
        try:
            model = MODEL_BUILDERS[self._args.model].build_from_variant(variant=self._args.variant)
        except (AttributeError, OSError) as e:
            self._logger.error(e)
            raise CLIException(f"Invalid variant '{self._args.variant}' for model") from e

        if self._args.collection == '':
            self._logger.info("Querying without RAG")
            rag = RAG(model=model)
            response = rag.query(self._args.query)
            context = []
        else:
            self._logger.info("Using collection '%s'", self._args.collection)
            storage = ChromaDBStorage(model=self._args.embedder, db_path=self._args.database_dir)
            rag = RAG(model=model, storage=storage)
            response, context = rag.query_with_documents(self._args.query, self._args.collection,
                                                         num_docs=10,
                                                         max_distance=self._args.max_distance)

        self.__show_response(response, context, self._args.context)

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
        self.parser.add_argument('--context',
                                 default=False,
                                 action='store_true',
                                 help='Show the relevant context passed to the LLM to answer the question')
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
                                 choices=MODEL_BUILDERS.keys(),
                                 help=f'''
                                    Base model to use as a conversational agent.
                                    Defaults to {DEFAULTS['model']}
                                    ''')
        self.parser.add_argument('--max-distance',
                                 default=1.0,
                                 type=float,
                                 help='Maximum cosine distance for a document to be considered relevant.')
        self.parser.add_argument('--variant',
                                 default='',
                                 type=str,
                                 help='''
                                    Variant of model. See HuggingFace list of models
                                    (https://huggingface.co/models). Ej: 4b-it
                                    ''')

        args = self.parser.parse_args()

        if args.query == '':
            raise CLIException("Please specify an input query.")

        if args.database_dir != '' and not os.path.exists(args.database_dir):
            raise CLIException(f"Database directory '{args.database_dir}' not found")

        if args.max_distance <= 0:
            raise CLIException("Invalid maximum distance")

        self._args = args

    def __show_response(self, response:str, context:list[Document], show_context:bool):
        if show_context:
            for doc in context:
                doc.print_to_console()

        print(response)

        self._logger.info("Response served. Length: %s", len(response))

if __name__ == "__main__":
    run_cli(CLIController)
