from typing import List
from abc import ABC, abstractmethod
import chromadb
import pandas as pd

class Storage(ABC):

    @abstractmethod
    def save_info(self, name:str, sentences:List[str], metadatas:List[str], embeddings:List[str]=None):
        pass

class ChromaDBStorage(Storage):

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./db")

    def save_info(self, name:str, sentences:List[str], metadatas:List[str], embeddings:List[str]=None):
        try:
            collection = self.client.get_collection(name)
            print("Collection already exists")
            return
        except chromadb.errors.NotFoundError as e:
            collection = self.client.create_collection(name)

        collection.add(
            documents=sentences,
            metadatas=metadatas,
            ids = [str(i+1) for i in range(len(sentences))]
        )

class CSVStorage(Storage):

    def save_info(self, name:str, sentences:List[str], metadatas:List[str], embeddings:List[str]=None):
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