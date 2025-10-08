"""Module to define classes to generate embeddings from sentences."""
from simplerag.singleton import Singleton
from abc import ABC, abstractmethod
from enum import Enum

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from llama_cpp import Llama

class Embedder(ABC):
    """Base class for embedding functions of different sources."""

    @abstractmethod
    def __call__(self, sentences: list[str]):
        """Return the embeddings."""

class EmbedderBuilder:
    """Factory class for different types of embedders."""

    @classmethod
    def get_from_type(cls, embedder_type:str, model_name:str, **model_args):
        """Get embedder from type."""
        try:
            return Embedders[embedder_type.upper()].value(model_name, **model_args)
        except KeyError:
            return None

    @classmethod
    def get_from_model_name(cls, model_name:str, **model_args):
        """Get embedder depending on the model name."""
        if model_name.startswith('sentence-transformers'):
            return STEmbedder(model_name, **model_args)
        elif model_name.endswith('.gguf'):
            return GGUFEmbedder(model_name, **model_args)
        else:
            return TREmbedder(model_name, **model_args)

class STEmbedder(Embedder):
    """Class to create embeddings using SentenceTransformers from HuggingFace."""
    __metaclass__ = Singleton

    def __init__(self, model_name:str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, sentences: list[str]):
        return self.model.encode(sentences, batch_size=1, convert_to_numpy=True).tolist()

class TREmbedder(Embedder):
    """Class to create embeddings using Transformers library."""
    __metaclass__ = Singleton

    def __init__(self, model_name:str = 'Qwen/Qwen3-Embedding-0.6B', device: str = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def __call__(self, sentences: list[str]):
        batch_size = 1
        max_length = 2048

        embeddings = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start+batch_size]
            # Tokenize the input texts
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
            sentence_embeddings = self.__last_token_pool(outputs.last_hidden_state,
                                                         batch_dict['attention_mask'])

            # normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.extend(sentence_embeddings.cpu().numpy().tolist())

        return embeddings

    def __last_token_pool(self, last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class GGUFEmbedder(Embedder):
    """Class to create embeddings from GGUF models using llama_cpp."""
    __metaclass__ = Singleton

    def __init__(self, model_name:str, device: str = 'cpu'):
        self.model = Llama(
            model_path=model_name,
            embedding=True,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )

    def __call__(self, sentences):
        embeddings = []
        for sentence in sentences:
            embeddings.append(self.model.embed(sentence))

        return embeddings

class Embedders(Enum):
    """Different types of embedders"""
    SENTENCE_TRANSFORMERS = STEmbedder
    TRANSFORMERS = TREmbedder
    GGUF = GGUFEmbedder