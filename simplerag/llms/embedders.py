"""Module to define classes to generate embeddings from sentences."""

from sentence_transformers import SentenceTransformer

from simplerag.singleton import Singleton

class STEmbedder(metaclass=Singleton):
    """Class to create embeddings using SentenceTransformers from HuggingFace."""

    def __init__(self, model:str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model)

    def get_embeddings(self, sentences):
        """Calculate and get the embeddings for a list of sentences."""
        embeddings = self.model.encode(sentences)

        return embeddings
