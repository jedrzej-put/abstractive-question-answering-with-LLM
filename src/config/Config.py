from dataclasses import dataclass, field
from pathlib import Path
from src.lib.VectorDb.AbstractVectorDb import AbstractVectorDb
from langchain_core.embeddings import Embeddings

@dataclass(frozen=True)
class Config:
    passages_path: Path
    vector_db: AbstractVectorDb
    vector_db_path: Path
    embedding_model: Embeddings
    max_chunk_size: int = 512
    chunk_overlap: int = 100
    split_into_chunks: bool = False
    text_splitter_separator: str = ""