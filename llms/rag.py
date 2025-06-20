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
