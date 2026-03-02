from dataclasses import dataclass

@dataclass
class Defaults:
    embedder: str = 'Qwen/Qwen3-Embedding-0.6B'
    model: str = 'Qwen/Qwen3-0.6B'
    database_dir: str = './db'
    extraction_type: str = 'data'
    inner_splitter: str = 'section'
    max_chars: int = 8000
    pdfplumber_raw: bool = False
    chat_num_related_docs: int = 5