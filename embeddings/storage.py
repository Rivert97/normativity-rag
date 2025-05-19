from abc import ABC, abstractmethod
import chromadb
import chromadb.errors
from chromadb.utils import embedding_functions
import pandas as pd

class Storage(ABC):

    @abstractmethod
    def save_info(self, name:str, sentences:list[str], metadatas:list[str], embeddings:list[str]=None):
        pass

    @abstractmethod
    def query_sentence(self, collection:str, sentence:str, n_results:int) -> list[dict]:
        pass

class ChromaDBStorage(Storage):

    def __init__(self, model:str='all-MiniLM-L6-v2', db_path:str='./db'):
        self.client = chromadb.PersistentClient(path=db_path)
        self.em_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)

    def save_info(self, name:str, sentences:list[str], metadatas:list[str], embeddings:list[str]=None):
        try:
            collection = self.client.get_collection(name, embedding_function=self.em_func)
        except chromadb.errors.NotFoundError as e:
            collection = self.client.create_collection(name, embedding_function=self.em_func)

        collection.add(
            documents=sentences,
            metadatas=metadatas,
            ids = [str(i+1) for i in range(len(sentences))]
        )

    def query_sentence(self, collection, sentence, n_results):
        import pdb; pdb.set_trace()
        try:
            collection = self.client.get_collection(collection, embedding_function=self.em_func)
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