"""Module to define clases that apply different types of rag."""

from .models import Model
from .storage import Storage

class RAG:
    """A class to perform basic RAG."""

    def __init__(self, model:Model, storage:Storage|None=None):
        self.model = model
        self.storage = storage

    def query(self, query:str, collection:str='', num_docs:int=5,
              max_distance:float=1.0, add_to_history:bool=True) -> str:
        """Retrieve an answer if a collection is specified it passes relevant
        documents as context."""
        if collection == '':
            return self.model.query(query, add_to_history), []

        if self.storage is None:
            return '', []

        documents = self.storage.query_sentence(collection, query, num_docs)

        relevant_docs = []
        for doc in documents:
            if doc.get_distance() > max_distance:
                continue

            relevant_docs.append(doc)

        if len(relevant_docs) > 0:
            response = self.model.query_with_documents(query, relevant_docs, add_to_history)
        else:
            response = self.model.query(query, add_to_history)

        return response, relevant_docs

    def batch_query(self, queries:list[str], collection:str='', num_docs:int=5,
                    max_distance:float=1.0) -> list[dict]:
        """Query multiple sentences to the model with or without context.

        Messages history is deleted between queries so each one is independent.
        """
        responses = []
        n_queries = len(queries)

        for _, query in enumerate(queries):
            response, docs = self.query(query, collection, num_docs, max_distance,
                                        add_to_history=False)
            d = {
                'response': response,
                'relevant_docs': docs
            }
            responses.append(d)

        print(f"Finished {n_queries}/{n_queries}")

        return responses
