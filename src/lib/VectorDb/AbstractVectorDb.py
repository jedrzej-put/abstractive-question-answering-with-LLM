import abc
import logging
from typing import Iterable, List, Optional, Any
from src.lib.common.tools import logger

class AbstractVectorDb:
    @abc.abstractmethod
    def load_vector_db(self)-> None:
        pass

    @abc.abstractmethod
    def store_embeddings(self, docs_embeddings_pairs: Iterable[tuple[str, List[float]]], metadatas: Optional[list[dict]] = None, logger: logging.Logger=logger,  **kwargs: Any):
        pass

    @abc.abstractmethod
    def similarity_search(self, query, n_docs=3) -> tuple[list[str], list[float]]:
        pass