"""Module to define clases that apply different types of rag."""

from .models import HFModel
from .storage import Storage

class RAG:
    """A class to perform basic RAG."""

    def __init__(self, model:HFModel, storage:Storage|None=None):
        self.model = model
        self.storage = storage

    def query(self, query:str) -> str:
        """Retrieve an answer with no additional context."""
        return self.model.query(query)

    def query_with_documents(self, query:str, collection:str) -> str:
        """Retrieve an answer by first searching relevant documents in the colleciton."""
        if self.storage is None:
            return []

        documents = self.storage.query_sentence(collection, query, 5)

        return self.model.query_with_documents(query, documents)
