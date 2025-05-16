from typing import List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

class Embedder(ABC):

    @abstractmethod
    def get_embeddings(self, sentences: List[str]):
        pass

class AllMiniLM(Embedder):

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, sentences):
        embeddings = self.model.encode(sentences)

        return embeddings