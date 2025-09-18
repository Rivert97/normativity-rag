"""Script to evaluate the performance of the RAG system.

This script requires a database previously created with get_embeddings.py script
and a dataset of questions and answers.
"""
import argparse

from rouge_score import rouge_scorer, scoring

from simplerag.llms.models import Builders
from simplerag.llms.rag import RAGQueryConfig
from .utils.controllers import run_cli
from .utils.eval_controllers import EvalCLI

PROGRAM_NAME = 'EvalRAG'
VERSION = '1.00.00'

class EvalRAGCLI(EvalCLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

    def run(self):
        """Run the script logic."""
        dataset = self.load_dataset(self._args.dataset)
        storage = self.get_storage(self._args.embedder, self._args.database_dir)
        model = self.load_model(self._args.model, self._args.variant)
        rag = self.get_rag(model, storage)

        self._logger.info("Getting Responses")
        train = dataset['train']
        query_config = RAGQueryConfig(
            collection=self._args.collection,
            num_docs=self._args.number_results,
            add_to_history=False
        )
        responses = rag.batch_query(train['question'], query_config)

        self._logger.info("Evaluating")
        rouge_score = self.__calculate_rouge([res['response'] for res in responses],
                               [answ['text'][0] for answ in train['answers']])

        print("ROUGE 1:", rouge_score['rouge1'])
        print("ROUGE 2:", rouge_score['rouge2'])
        print("ROUGE L:", rouge_score['rougeL'])
        print("ROUGE LSum:", rouge_score['rougeLsum'])
        self._logger.info("ROUGUE: %s", rouge_score)

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument('-m', '--model',
                                 default='GEMMA',
                                 type=str,
                                 choices=[b.name for b in Builders],
                                 help='''
                                    Base model to use as a conversational agent.
                                    Defaults to GEMMA.
                                    ''')
        self.parser.add_argument('--variant',
                                 default='',
                                 type=str,
                                 help='''
                                    Variant of model. See HuggingFace list of models
                                    (https://huggingface.co/models). Ej: 4b-it
                                    ''')

        self._args = self.parser.parse_args()

    def __calculate_rouge(self, responses, answers):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"],
                                          use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for pred, ref in zip(responses, answers):
            score = scorer.score(ref, pred)
            aggregator.add_scores(score)
        result = aggregator.aggregate()

        return {metric: value.mid.fmeasure for metric, value in result.items()}

def main():
    """Run the script."""
    run_cli(EvalRAGCLI)

if __name__ == "__main__":
    run_cli(EvalRAGCLI)
