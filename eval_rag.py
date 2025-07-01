"""Script to evaluate the performance of the RAG system.

This script requires a database previously created with get_embeddings.py script
and a dataset of questions and answers.
"""
import argparse
import os

from datasets import load_dataset
import evaluate

from utils.controllers import CLI, run_cli
from utils.exceptions import CLIException
from llms.storage import ChromaDBStorage
from llms.rag import RAG
from llms.models import GemmaBuilder, LlamaBuilder, QwenBuilder, MistralBuilder

MODEL_BUILDERS = {
    'gemma': GemmaBuilder,
    'llama': LlamaBuilder,
    'qwen': QwenBuilder,
    'mistral': MistralBuilder,
}

PROGRAM_NAME = 'EvalRAG'
VERSION = '1.00.00'

class CLIController(CLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

        self._args = None

    def run(self):
        """Run the script logic."""
        self._logger.info('Loading dataset')
        dataset = load_dataset(self._args.dataset)

        self._logger.info('Loading database')
        storage = ChromaDBStorage(self._args.embedder, self._args.database_dir)

        self._logger.info("Loading Model '%s (%s)'", self._args.model, self._args.variant)
        try:
            model = MODEL_BUILDERS[self._args.model].build_from_variant(variant=self._args.variant)
        except (AttributeError, OSError) as e:
            self._logger.error(e)
            raise CLIException(f"Invalid variant '{self._args.variant}' for model") from e

        self._logger.info("Creating RAG")
        rag = RAG(model=model, storage=storage)

        self._logger.info("Getting Responses")
        train = dataset['train'][:2]
        responses = rag.batch_query(train['question'], self._args.collection, self._args.number_results,
                                    self._args.max_distance, independent_queries=True)

        self._logger.info("Evaluating")
        rouge_score = self.__calculate_ROUGE([res['response'] for res in responses],
                               [answ['text'][0] for answ in train['answers']])

        print("ROUGE 1:", rouge_score['rouge1'])
        print("ROUGE 2:", rouge_score['rouge2'])
        print("ROUGE L:", rouge_score['rougeL'])
        print("ROUGE LSum:", rouge_score['rougeLsum'])

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
        self.parser.add_argument('-m', '--model',
                                 default='gemma',
                                 type=str,
                                 choices=MODEL_BUILDERS.keys(),
                                 help=f'''
                                    Base model to use as a conversational agent.
                                    Defaults to gemma.
                                    ''')
        self.parser.add_argument('--max-distance',
                                 default=1.0,
                                 type=float,
                                 help='''
                                    Maximum cosine distance for a document to be considered
                                    relevant.
                                    ''')
        self.parser.add_argument('-n', '--number-results',
                                default=5,
                                type=int,
                                help='Number of relevant documents to retrieve. Defaults to 5')
        self.parser.add_argument('-s', '--split',
                                 default='train',
                                 type=str,
                                 help='Dataset split to be used to evaluate.')
        self.parser.add_argument('--variant',
                                 default='',
                                 type=str,
                                 help='''
                                    Variant of model. See HuggingFace list of models
                                    (https://huggingface.co/models). Ej: 4b-it
                                    ''')

        args = self.parser.parse_args()

        if args.collection == '':
            raise CLIException("Please specify a collection")

        if not os.path.exists(args.database_dir):
            raise CLIException(f"Database folder '{args.database_dir}' not found")

        if args.max_distance <= 0:
            raise CLIException("Invalid maximum distance")

        self._args = args

    def __calculate_ROUGE(self, responses, answers):
        rouge = evaluate.load('rouge')
        score = rouge.compute(predictions=responses,
                              references=answers)

        return score

if __name__ == "__main__":
    run_cli(CLIController)
