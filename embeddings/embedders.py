from sentence_transformers import SentenceTransformer

class STEmbedder():

    def __init__(self, model:str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model)

    def get_embeddings(self, sentences):
        embeddings = self.model.encode(sentences)

        return embeddings