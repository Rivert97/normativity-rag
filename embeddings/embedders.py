"""Module to define classes to generate embeddings from sentences."""

from sentence_transformers import SentenceTransformer

class STEmbedder():
    """Class to create embeddings using SentenceTransformers from HuggingFace."""

    def __init__(self, model:str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model)

    def get_embeddings(self, sentences):
        """Calculate and get the embeddings for a list of sentences."""
        embeddings = self.model.encode(sentences)

        return embeddings
