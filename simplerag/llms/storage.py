"""Module to define classes to store embeddings in different ways."""

import os
from abc import ABC, abstractmethod

import chromadb
import chromadb.errors
import pandas as pd

from .data import Document
from .embedders import EmbedderBuilder

class Storage(ABC):
    """Interface that defines methods that should be implemented by a storage."""

    @abstractmethod
    def save_info(self, name:str, info: dict[str, list[str]], id_prefix: str = ''):
        """Save information into the corresponding storage."""

    @abstractmethod
    def query_sentence(self, collection:str, sentence:str, n_results:int) -> tuple[list[Document], int]:
        """Find related documents using the corresponding storage."""

class ChromaDBStorage(Storage):
    """Class to store data in a chromadb database."""

    def __init__(self, model:str='all-MiniLM-L6-v2', db_path:str='./db', device:str = 'cuda'):
        self.model = model
        self.device = device
        self.client = chromadb.PersistentClient(path=db_path)
        self.hnsw_space = "ip" if 'dot' in model else 'cosine'

    def save_info(self, name:str, info: dict[str, list[str]], id_prefix: str = ''):
        """Save the provided data into a collection inside a chromadb database."""
        em_func = EmbedderBuilder.get_from_model_name(self.model, device=self.device)
        try:
            collection = self.client.get_collection(name, embedding_function=em_func)
        except chromadb.errors.NotFoundError:
            collection = self.client.create_collection(name,
                                                       embedding_function=em_func,
                                                       configuration={
                                                           "hnsw": {
                                                               "space": self.hnsw_space,
                                                           }
                                                       })

        collection.add(
            documents=info.get('sentences'),
            metadatas=info.get('metadatas'),
            ids = [f'{id_prefix}{i+1}' for i in range(len(info.get('sentences')))]
        )

    def query_sentence(self, collection, sentence, n_results) -> tuple[list[Document], int]:
        """Make a query to the database to find similar sentences."""
        em_func = EmbedderBuilder.get_from_model_name(self.model, device=self.device)
        try:
            chromadb_collection = self.client.get_collection(collection,
                                                             embedding_function=em_func)
        except (chromadb.errors.NotFoundError, ValueError) as e:
            print(e)
            return [], 0

        return self.__query(chromadb_collection, sentence, n_results, em_func)

    def batch_query(self, collection, sentences, n_results) -> list[list[Document]]:
        """Make multiple queries to the database to find similar sentences."""
        em_func = EmbedderBuilder.get_from_model_name(self.model, device=self.device)
        try:
            chromadb_collection = self.client.get_collection(collection,
                                                             embedding_function=em_func)
        except (chromadb.errors.NotFoundError, ValueError) as e:
            print(e)
            return []

        results = []
        for sentence in sentences:
            documents, _ = self.__query(chromadb_collection, sentence, n_results, em_func)
            results.append(documents)

        return results

    def get_all_from_parent(self, collection, document_name, parent) -> list[Document]:
        """Get all documents from a document and a specific parent."""
        em_func = EmbedderBuilder.get_from_model_name(self.model, device=self.device)
        try:
            chromadb_collection = self.client.get_collection(collection,
                                                             embedding_function=em_func)
        except (chromadb.errors.NotFoundError, ValueError) as e:
            print(e)
            return []

        results = chromadb_collection.get(
            where={
                '$and': [
                    {'document_name': document_name},
                    {'parent': parent.split('/')[-1]},
                ]
            }
        )

        if not results['ids']:
            return []

        documents = []
        for i, _ in enumerate(results['ids']):
            doc = Document(
                content=results['documents'][i],
                metadata=results['metadatas'][i],
            )
            documents.append(doc)

        return documents

    def __query(self, chromadb_collection, sentence, n_results, em_func) -> tuple[list[Document], int]:
        results = chromadb_collection.query(
            query_texts=[sentence],
            n_results=n_results,
            include=['documents', 'metadatas', 'embeddings', 'distances'],
        )
        token_count = em_func.get_last_token_count()

        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                embeddings=results['embeddings'][0][i],
                distance=results['distances'][0][i],
            )
            documents.append(doc)

        return documents, token_count

class CSVStorage(Storage):
    """Class to store embeddings and in a CSV file."""

    def save_info(self, name:str, info: dict[str, list[str]], id_prefix: str = ''):
        """
        Save the provided data into a CSV file for the embeddings and another
        file for everything else.
        """
        basepath, basename = os.path.split(name)
        basename = os.path.splitext(basename)[0]
        df = pd.DataFrame(info.get('embeddings'))
        df.to_csv(os.path.join(basepath, f"{basename}_embeddings.csv"), sep=',', index=True)

        df = pd.DataFrame(info.get('metadatas'))
        df['sentences'] = info.get('sentences')
        df.to_csv(f"{name}.csv", sep=',', index=True)

    def query_sentence(self, collection:str, sentence:str, n_results:int) -> tuple[list[dict], int]:
        """Find similar sentences. Not implemented."""
        return None, 0
