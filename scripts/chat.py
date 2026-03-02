"""Script to perform RAG with a custom database."""

import argparse
import os

from simplerag.llms.storage import ChromaDBStorage
from simplerag.llms.rag import RAG, RAGQueryConfig
from simplerag.llms.models import ModelBuilder
from simplerag.llms.data import Document
from .utils.controllers import CLI, run_cli
from .utils.exceptions import CLIException
from .utils.defaults import Defaults

DEFAULTS = {
    'prompt_file': './prompts/system.txt',
}

PROGRAM_NAME = 'chat'
VERSION = '1.00.00'

class CLIChatController(CLI):
    """This class controls the execution of the program when used as CLI."""
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

        self._args = None
        self.system_prompt = None

    def run(self):
        """Run the script logic."""
        self._logger.info("Loading Model '%s'", self._args.model)
        model = ModelBuilder.get_from_model_name(
            self._args.model,
            system_prompt=self.system_prompt
        )

        if model is None:
            raise CLIException(f"Invalid model '{self._args.model}'")

        if self._args.collection == '':
            self._logger.info("Querying without RAG")
            rag = RAG(model=model)
        else:
            storage = ChromaDBStorage(model=self._args.embedder, db_path=self._args.database_dir,
                                      device='cpu')
            rag = RAG(model=model, storage=storage)

        if self._args.query == '':
            self.__process_interactive(rag)
        else:
            query_config = RAGQueryConfig(
                collection=self._args.collection,
                num_docs=self._args.num_docs,
            )
            response, context = rag.query(self._args.query, query_config)

            self.__show_response(response, context, self._args.show_context)

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('-c', '--collection',
                                 default='',
                                 type=str,
                                 help='Name of the collection to use. Must exist in the database')
        self.parser.add_argument('--show-context',
                                 default=False,
                                 action='store_true',
                                 help='''
                                    Show the relevant context passed to the LLM to answer
                                    the question.
                                    ''')
        self.parser.add_argument('-d', '--database-dir',
                                 default=Defaults.database_dir,
                                 type=str,
                                 help=f'''
                                    Directory where the database is stored.
                                    Defaults to {Defaults.database_dir}
                                    ''')
        self.parser.add_argument('-e', '--embedder',
                                 default=Defaults.embedder,
                                 type=str,
                                 help=f'''
                                    Embeddings model to be used. Must match the database embedder.
                                    Defaults to {Defaults.embedder}
                                    ''')
        self.parser.add_argument('-m', '--model',
                                 default=Defaults.model,
                                 type=str,
                                 help=f'''
                                    Model to use as a conversational agent.
                                    Defaults to {Defaults.model}
                                    ''')
        self.parser.add_argument('-n', '--num-docs',
                                 default=Defaults.chat_num_related_docs,
                                 type=int,
                                 help=f'''
                                    Number of context documents used to answer the question.
                                    Defaults to {Defaults.chat_num_related_docs}
                                    ''')
        self.parser.add_argument('-p', '--prompt-file',
                                 default=DEFAULTS['prompt_file'],
                                 type=str,
                                 help=f'''File of a custom system prompt to be passed to the model.
                                    Defaults to {DEFAULTS['prompt_file']}''')
        self.parser.add_argument('--query',
                                 default='',
                                 type=str,
                                 help='Sentence query to be answered by the model')

        args = self.parser.parse_args()

        if args.database_dir != '' and not os.path.exists(args.database_dir):
            raise CLIException(f"Database directory '{args.database_dir}' not found")

        if args.num_docs < 0:
            raise CLIException("Invalid number of context documents")

        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()

        self._args = args

    def __process_interactive(self, rag=RAG):
        self._logger.info("Loading interactive mode")

        print("Bienvenido al ChatBot UG. Presione Ctrl+c para salir\n")
        while True:
            try:
                query = input(">> ")
            except KeyboardInterrupt:
                print("\nGracias")
                break

            if query == '':
                continue

            query_config = RAGQueryConfig(
                collection=self._args.collection,
                num_docs=self._args.num_docs,
            )
            response, context = rag.query(query, query_config)

            self.__show_response(response, context, self._args.show_context)

    def __show_response(self, response:str, context:list[Document], show_context:bool):
        if show_context:
            for doc in context:
                doc.print_to_console()

        print(response)

        self._logger.info("Response served. Length: %s", len(response))

def main():
    """Run the script."""
    run_cli(CLIChatController)

if __name__ == "__main__":
    run_cli(CLIChatController)
