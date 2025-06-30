"""Script to evaluate the performance of the retrieval step.

This script requires a database previously created with get_embeddings.py script
and a dataset of questions and answers.
"""
import argparse
import os

from datasets import load_dataset
import numpy as np

from utils.controllers import CLI, run_cli
from utils.exceptions import CLIException
from llms.storage import ChromaDBStorage

PROGRAM_NAME = 'EvalRetrieval'
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

        self._logger.info('Querying sentences')
        questions = dataset['train']
        results = storage.batch_query(self._args.collection, questions['question'],
                                     self._args.number_results)

        precision, recall, f1 = self.__calculate_metrics_at_k(questions, results, storage,
                                                              self._args.number_results)

        print("N Samples:", len(questions))
        print(f"Precision@{self._args.number_results}:{precision}")
        print(f"Recall@{self._args.number_results}:{recall}")
        print(f"F1@{self._args.number_results}:{f1}")

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

        args = self.parser.parse_args()

        if args.collection == '':
            raise CLIException("Please specify a collection")

        if not os.path.exists(args.database_dir):
            raise CLIException(f"Database folder '{args.database_dir}' not found")

        self._args = args

    def __calculate_metrics_at_k(self, questions, queries_results:list, storage:ChromaDBStorage,
                                 k:int) -> tuple[float, float, float]:
        metrics = {
            'precisions': np.zeros((len(questions),), dtype=np.float32),
            'recalls': np.zeros((len(questions),), dtype=np.float32),
            'f1s': np.zeros((len(questions),), dtype=np.float32),
        }

        for q, question in enumerate(questions):
            n_relevant = 0
            all_relevant = storage.get_all_from_parent(self._args.collection, question['title'],
                                                       question['context'])
            for document in queries_results[q]:
                if self.__is_relevant(question, document):
                    n_relevant += 1

            precision = n_relevant/k
            recall = n_relevant/len(all_relevant) if len(all_relevant) > 0 else 0
            f1 = (2*precision*recall)/(precision+recall) if precision+recall > 0 else 0

            metrics['precisions'][q] = precision
            metrics['recalls'][q] = recall
            metrics['f1s'][q] = f1

        return metrics['precisions'].mean(), metrics['recalls'].mean(), metrics['f1s'].mean()

    def __is_relevant(self, question, document):
        if question['title'] != document.metadata['document_name']:
            return False

        article = question['context'].lower().strip()
        if not document.metadata['path'].lower().endswith(article):
            return False

        return True


if __name__ == "__main__":
    run_cli(CLIController)
