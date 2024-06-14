from langchain_community.vectorstores.faiss import FAISS
from functools import lru_cache
import faiss
import logging
from typing import Iterable, List, Optional, Any
from src.lib.VectorDb.AbstractVectorDb import AbstractVectorDb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

class FAISSVectorDb(AbstractVectorDb):
    def __init__(self, embedding_model, vector_db_path):
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.flag_loaded_vector_db = False

    @lru_cache(maxsize=1)
    def load_vector_db(self) -> None:
        if self.vector_db_path.exists():
            self.vector_db = FAISS.load_local(self.vector_db_path, self.embedding_model)
            logger.info(f"Loaded vector db from {self.vector_db_path}")
        else:
            self.vector_db = FAISS.from_texts(["Init"], self.embedding_model, [{"id":-1, "title":"Init"}])
            logger.info(f"Created new vector db at {self.vector_db_path}")
        self.flag_loaded_vector_db = True
    

    def store_embeddings(self, docs_embeddings_pairs: Iterable[tuple[str, List[float]]], metadatas: Optional[list[dict]] = None, logger: logging.Logger=logger, **kwargs: Any):
        if not self.flag_loaded_vector_db:
            self.load_vector_db()

        num_indexes_in_db = self.vector_db.index.ntotal

        new_ids = self.vector_db.add_embeddings(docs_embeddings_pairs, metadatas)
        self.vector_db.save_local(self.vector_db_path)
        
        logger.info(f"vectors in db: {self.vector_db.index.ntotal}, new indexex: {self.vector_db.index.ntotal - num_indexes_in_db}, len new ids: {len(new_ids)}")

    def similarity_search(self, query, n_docs=3) -> tuple[list[str], list[float]]:
        if not self.flag_loaded_vector_db:
            self.load_vector_db()

        docs_scores = self.vector_db.similarity_search_with_score(query, k=n_docs)
        docs, scores=list(zip(*docs_scores))
        return docs, scores

