from functools import lru_cache
from langchain_core.embeddings import Embeddings
from src.config.Config import Config

@lru_cache(maxsize=1)
def get_embedding_model(config: Config) -> Embeddings:
    model_cls = config.embedding_model
    return model_cls()