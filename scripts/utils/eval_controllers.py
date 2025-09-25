"""Module to define controllers for evaluationg the system."""
import os
import argparse

from datasets import load_dataset

from simplerag.llms.storage import ChromaDBStorage
from simplerag.llms.models import ModelBuilder, Model
from simplerag.llms.rag import RAG
from .controllers import CLI
from .exceptions import CLIException

class EvalCLI(CLI):
    """Special controller for CLI in evaluation mode."""
    def __init__(self, program_name:str, documentation:str, version:str):
        super().__init__(program_name, documentation, version)

        self._args = None

    def load_dataset(self, dataset:str):
        """Load the dataset for evluation."""
        self._logger.info('Loading dataset')
        dataset = load_dataset(dataset)

        return dataset

    def get_storage(self, embedder:str, database_dir:str) -> ChromaDBStorage:
        """Load the database of embeddings."""
        self._logger.info('Loading database')
        storage = ChromaDBStorage(embedder, database_dir)

        return storage

    def load_model(self, model_id:str) -> Model:
        """Load the LLM model"""
        self._logger.info("Loading Model '%s'", model_id)
        model = ModelBuilder.get_from_id(model_id)
        if model is None:
            raise CLIException(f"Invalid model '{model_id}'")

        return model

    def get_rag(self, model:Model, storage:ChromaDBStorage) -> RAG:
        """Get the rag interface."""
        self._logger.info("Creating RAG")
        rag = RAG(model=model, storage=storage)

        return rag

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('dataset',
                                default='Rivert97/ug-normativity',
                                type=str,
                                help='''
                                    Name of the HuggingFace dataset to be used.
                                    Defaults to Rivert97/ug-normativity
                                    ''')
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
        self.parser.add_argument('-s', '--split',
                                 default='train',
                                 type=str,
                                 help='Dataset split to be used to evaluate.')

    def eval_args(self):
        super().eval_args()

        if self._args.collection == '':
            raise CLIException("Please specify a collection")

        if not os.path.exists(self._args.database_dir):
            raise CLIException(f"Database folder '{self._args.database_dir}' not found")
