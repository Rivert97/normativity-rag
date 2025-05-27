"""Module to define clases that apply different types of rag."""

from .models import LLMModel
from .storage import Storage

class RAG:
    """A class to perform basic RAG."""

    def __init__(self, model:LLMModel, storage:Storage):
        self.model = model
        self.storage = storage

    def query(self, query:str) -> str:
        """Retrieve an answer with no additional context."""
        return self.model.query(query)

    def query_with_documents(self, query:str, collection:str) -> str:
        """Retrieve an answer by first searching relevant documents in the colleciton."""
        documents = self.storage.query_sentence(collection, query, 5)
        doc_contents = self.__prepare_documents(documents)

        return self.model.query_with_documents(query, doc_contents)

    def __prepare_documents(self, raw_documents:list[str]):
        documents = []
        for d in raw_documents:
            doc = {
                'title': d['metadata']['title'],
                'text': d['content'],
            }
            documents.append(doc)

        return documents
