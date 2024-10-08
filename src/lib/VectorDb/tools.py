from functools import lru_cache
from src.config.Config import Config
from src.lib.VectorDb.AbstractVectorDb import AbstractVectorDb
from src.lib.Embeddings.tools import get_embedding_model
@lru_cache(maxsize=1)
def get_vector_db(config: Config) -> AbstractVectorDb:
    vector_db_cls = config.vector_db
    return vector_db_cls(
        get_embedding_model(config), 
        config.vector_db_path,
    )

