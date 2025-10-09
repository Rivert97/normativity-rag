"""Module to define classes to generate embeddings from sentences."""
from abc import abstractmethod
from enum import Enum

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from llama_cpp import Llama

from .singleton import Singleton

# pylint: disable=redefined-builtin

class Embedder():
    """Base class for embedding functions of different sources."""

    @abstractmethod
    def __call__(self, input: list[str]):
        """
        Return the embeddings.
        input param needs to be called like that to use it with ChromaDB.
        """

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

        if model_name.endswith('.gguf'):
            return GGUFEmbedder(model_name, **model_args)

        return TREmbedder(model_name, **model_args)

class STEmbedder(Embedder, metaclass=Singleton):
    """Class to create embeddings using SentenceTransformers from HuggingFace."""

    def __init__(self, model_name:str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        """Initialize a sentence_transformer embedder."""
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: list[str]):
        """Get the embeddings."""
        return self.model.encode(input, batch_size=1, convert_to_numpy=True).tolist()

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function."""
        return "sentence_transformer"

class TREmbedder(Embedder, metaclass=Singleton):
    """Class to create embeddings using Transformers library."""

    def __init__(self, model_name:str = 'Qwen/Qwen3-Embedding-0.6B', device: str = 'cpu'):
        """Initialize the transformers model to obtain embeddings."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def __call__(self, input: list[str]):
        """Get the embeddings."""
        batch_size = 1
        max_length = 2048

        embeddings = []
        for start in range(0, len(input), batch_size):
            batch = input[start:start+batch_size]
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

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        batch_range = torch.arange(batch_size, device=last_hidden_states.device)
        return last_hidden_states[batch_range, sequence_lengths]

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function."""
        return "transformer"

class GGUFEmbedder(Embedder, metaclass=Singleton):
    """Class to create embeddings from GGUF models using llama_cpp."""

    def __init__(self, model_name:str):
        """Initialize the llama_cpp model to obtain embeddings."""
        self.model = Llama(
            model_path=model_name,
            embedding=True,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )

    def __call__(self, input):
        """Get the embeddings."""
        embeddings = []
        for sentence in input:
            embeddings.append(self.model.embed(sentence))

        return embeddings

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function."""
        return "llama_cpp"

class Embedders(Enum):
    """Different types of embedders"""
    SENTENCE_TRANSFORMERS = STEmbedder
    TRANSFORMERS = TREmbedder
    GGUF = GGUFEmbedder
