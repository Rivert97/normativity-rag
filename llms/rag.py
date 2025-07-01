"""Module to define clases that apply different types of rag."""

from .models import Model
from .storage import Storage

class RAG:
    """A class to perform basic RAG."""

    def __init__(self, model:Model, storage:Storage|None=None):
        self.model = model
        self.storage = storage

    def query(self, query:str, collection:str='', num_docs:int=5,
                             max_distance:float=1.0) -> str:
        """Retrieve an answer if a collection is specified it passes relevant
        documents as context."""
        if collection == '':
            return self.model.query(query), []

        if self.storage is None:
            return '', []

        documents = self.storage.query_sentence(collection, query, num_docs)

        relevant_docs = []
        for doc in documents:
            if doc.get_distance() > max_distance:
                continue

            relevant_docs.append(doc)

        if len(relevant_docs) > 0:
            response = self.model.query_with_documents(query, relevant_docs)
        else:
            response = self.model.query(query)

        return response, relevant_docs

    def batch_query(self, queries:list[str], collection:str='', num_docs:int=5,
                    max_distance:float=1.0, verbose:bool=True,
                    independent_queries:bool=False) -> list[dict]:
        """Query multiple sentences to the model with or without context."""
        responses = []
        n_queries = len(queries)

        for i, query in enumerate(queries):
            if verbose:
                print(f"Querying {i+1}/{n_queries}", end='\r')
            response, docs = self.query(query, collection, num_docs, max_distance)
            d = {
                'response': response,
                'relevant_docs': docs
            }
            responses.append(d)

            if independent_queries:
                self.model.init_messages()
        else:
            print(f"Finished {n_queries}/{n_queries}")

        return responses
