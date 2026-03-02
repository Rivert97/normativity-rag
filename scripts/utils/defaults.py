"""Module to handle the default values of the application."""

from dataclasses import dataclass

DEFAULT_PARSE_PARAMS = {
    'pdf_margins': {
        'top': 0.1,
        'bottom': 0.95,
        'left': 0.05,
        'right': 0.95,
    }
}

@dataclass
class Defaults:
    """Default values for the application."""
    embedder: str = 'Qwen/Qwen3-Embedding-0.6B'
    model: str = 'Qwen/Qwen3-0.6B'
    database_dir: str = './db'
    extraction_type: str = 'data'
    inner_splitter: str = 'section'
    max_chars: int = 8000
    chat_num_related_docs: int = 5

@dataclass
class DefaultInferenceParams:
    """Default values for inference."""
    prompt_file: str = './prompts/system.txt'
    model_context: int = 8192
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.1
