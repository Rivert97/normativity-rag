"""Module to define classes to store embeddings in different ways."""

from abc import ABC, abstractmethod
import chromadb
import chromadb.errors
from chromadb.utils import embedding_functions
import pandas as pd

from .data import Document

class Storage(ABC):
    """Interface that defines methods that should be implemented by a storage."""

    @abstractmethod
    def save_info(self, name:str, sentences:list[str], metadatas:list[str],
                  embeddings:list[str]=None):
        """Save information into the corresponding storage."""

    @abstractmethod
    def query_sentence(self, collection:str, sentence:str, n_results:int) -> list[dict]:
        """Find related documents using the corresponding storage."""

class ChromaDBStorage(Storage):
    """Class to store data in a chromadb database."""

    def __init__(self, model:str='all-MiniLM-L6-v2', db_path:str='./db'):
        self.client = chromadb.PersistentClient(path=db_path)
        self.em_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)

    def save_info(self, name:str, sentences:list[str], metadatas:list[str],
                  embeddings:list[str]=None):
        """Save the provided data into a collection inside a chromadb database."""
        try:
            collection = self.client.get_collection(name, embedding_function=self.em_func)
        except chromadb.errors.NotFoundError:
            collection = self.client.create_collection(name, embedding_function=self.em_func)

        collection.add(
            documents=sentences,
            metadatas=metadatas,
            ids = [str(i+1) for i in range(len(sentences))]
        )

    def query_sentence(self, collection, sentence, n_results):
        """Make a query to the database to find similar sentences."""
        try:
            collection = self.client.get_collection(collection, embedding_function=self.em_func)
        except (chromadb.errors.NotFoundError, ValueError) as e:
            print(e)
            return []

        results = collection.query(
            query_texts=[sentence],
            n_results=n_results,
            include=['documents', 'metadatas', 'embeddings', 'distances'],
        )

        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                embeddings=results['embeddings'][0][i],
                distance=results['distances'][0][i],
            )
            documents.append(doc)

        return documents

class CSVStorage(Storage):
    """Class to store embeddings in a CSV file."""

    def save_info(self, name:str, sentences:list[str], metadatas:list[str],
                  embeddings:list[str]=None):
        """Save the provided data into a CSV file."""
        df_dict = {
            'sentences': sentences,
            'metadatas': metadatas,
        }
        for dim in range(len(embeddings[0])):
            col_name = f'emb{dim}'
            df_dict[col_name] = []
            for e in embeddings:
                df_dict[col_name].append(e[dim])

        df = pd.DataFrame(df_dict)

        df.to_csv(name, sep=',', index=False)

    def query_sentence(self, collection:str, sentence:str, n_results:int) -> list[dict]:
        """Find similar sentences. Not implemented."""
        return None
