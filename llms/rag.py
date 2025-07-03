"""Module to define clases that apply different types of rag."""
from dataclasses import dataclass

from .models import Model
from .storage import Storage

@dataclass
class RAGQueryConfig:
    """Configuration parameters for making a RAG Query."""
    collection: str = ''
    num_docs: int = 5
    max_distance: float = 1.0
    add_to_history: bool = True

class RAG:
    """A class to perform basic RAG."""

    def __init__(self, model:Model, storage:Storage|None=None):
        self.model = model
        self.storage = storage

    def query(self, query:str, query_config: RAGQueryConfig) -> str:
        """Retrieve an answer if a collection is specified it passes relevant
        documents as context."""
        if query_config.collection == ''or self.storage is None:
            return self.model.query(query, query_config.add_to_history), []

        documents = self.storage.query_sentence(query_config.collection,
                                                query, query_config.num_docs)

        relevant_docs = []
        for doc in documents:
            if doc.get_distance() > query_config.max_distance:
                continue

            relevant_docs.append(doc)

        if len(relevant_docs) > 0:
            response = self.model.query_with_documents(query, relevant_docs,
                                                       query_config.add_to_history)
        else:
            response = self.model.query(query, query_config.add_to_history)

        return response, relevant_docs

    def batch_query(self, queries:list[str], query_config: RAGQueryConfig) -> list[dict]:
        """Query multiple sentences to the model with or without context."""
        responses = []
        n_queries = len(queries)

        for _, query in enumerate(queries):
            response, docs = self.query(query, query_config)
            d = {
                'response': response,
                'relevant_docs': docs
            }
            responses.append(d)

        print(f"Finished {n_queries}/{n_queries}")

        return responses

    def query_with_conversation(self, messages:list[dict[str,str]],
                                query_config:RAGQueryConfig) -> list[dict]:
        """Makes a RAG query with a full conversation."""
        if query_config.collection == '' or self.storage is None:
            return self.model.query_with_conversation(messages), []

        last_query = messages[-1]['content']
        documents = self.storage.query_sentence(query_config.collection, last_query,
                                                query_config.num_docs)

        relevant_docs = []
        for doc in documents:
            if doc.get_distance() > query_config.max_distance:
                continue

            relevant_docs.append(doc)

        if len(relevant_docs) > 0:
            response = self.model.query_with_conversation_and_documents(messages,
                                                                        relevant_docs)
        else:
            response = self.model.query_with_conversation(messages)

        return response, relevant_docs
