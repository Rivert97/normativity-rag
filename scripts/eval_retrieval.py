"""Script to evaluate the performance of the retrieval step.

This script requires a database previously created with get_embeddings.py script
and a dataset of questions and answers.
"""
import argparse

import numpy as np

from simplerag.llms.storage import ChromaDBStorage
from .utils.controllers import run_cli
from .utils.eval_controllers import EvalCLI

PROGRAM_NAME = 'EvalRetrieval'
VERSION = '1.00.00'

class EvalRetrievalCLI(EvalCLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

    def run(self):
        """Run the script logic."""
        dataset = self.load_dataset(self._args.dataset)
        storage = self.get_storage(self._args.embedder, self._args.database_dir)

        self._logger.info('Querying sentences')
        if self._args.file != '':
            dataset = dataset.filter(lambda row: row['title'] == self._args.file)

        questions = dataset['train']
        results = storage.batch_query(self._args.collection, questions['question'],
                                     self._args.number_results)

        precision, recall, f1 = self.__calculate_metrics_at_k(questions, results, storage,
                                                              self._args.number_results)

        print("N Samples:", len(questions))
        print(f"Precision@{self._args.number_results}:{precision}")
        print(f"Recall@{self._args.number_results}:{recall}")
        print(f"F1@{self._args.number_results}:{f1}")
        self._logger.info("N Samples: %d", len(questions))
        self._logger.info("Precision@%d: %f", self._args.number_results, precision)
        self._logger.info("Recall@%d: %f", self._args.number_results, recall)
        self._logger.info("F1@%d: %f", self._args.number_results, f1)

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('-f', '--file',
                                 default='',
                                 type=str,
                                 help='''
                                    Set to filter the database by file (title). Default ''
                                    ''')

        self._args = self.parser.parse_args()

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

def main():
    """Run the script."""
    run_cli(EvalRetrievalCLI)

if __name__ == "__main__":
    run_cli(EvalRetrievalCLI)
