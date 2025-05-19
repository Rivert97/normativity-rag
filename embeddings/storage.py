from abc import ABC, abstractmethod
import chromadb
import chromadb.errors
import pandas as pd

class Storage(ABC):

    @abstractmethod
    def save_info(self, name:str, sentences:list[str], metadatas:list[str], embeddings:list[str]=None):
        pass

    @abstractmethod
    def query_sentence(self, collection:str, sentence:str, n_results:int) -> list[dict]:
        pass

class ChromaDBStorage(Storage):

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./db")

    def save_info(self, name:str, sentences:list[str], metadatas:list[str], embeddings:list[str]=None):
        try:
            collection = self.client.get_collection(name)
        except chromadb.errors.NotFoundError as e:
            collection = self.client.create_collection(name)

        collection.add(
            documents=sentences,
            metadatas=metadatas,
            ids = [str(i+1) for i in range(len(sentences))]
        )

    def query_sentence(self, collection, sentence, n_results):
        try:
            collection = self.client.get_collection(collection)
        except chromadb.errors.NotFoundError as e:
            return None

        results = collection.query(
            query_texts=[sentence],
            n_results=n_results
        )

        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
            }
            documents.append(doc)

        return documents

class CSVStorage(Storage):

    def save_info(self, name:str, sentences:list[str], metadatas:list[str], embeddings:list[str]=None):
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
        return super().query_sentence(collection, sentence, n_results)