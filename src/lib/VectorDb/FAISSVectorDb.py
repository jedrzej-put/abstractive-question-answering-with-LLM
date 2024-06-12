from langchain_community.vectorstores.faiss import FAISS
from functools import lru_cache
import faiss
import logging
from src.lib.VectorDb.AbstractVectorDb import AbstractVectorDb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

class FAISSVectorDb(AbstractVectorDb):
    def __init__(self, embedding_model, vector_db_path):
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.flag_loaded_vector_db = False

    @lru_cache(maxsize=1)
    def load_vector_db(self):
        vector_db = FAISS.load_local(self.vector_db_path, self.embedding_model)
        logger.info(f"Loaded vector db from {self.vector_db_path}")
        return vector_db

    def store_embeddings(self, docs_embeddings_pairs, metadata):
        if self.vector_db_path.exists():
            index = faiss.read_index(self.vector_db_path / "index.faiss")
            num_indexes_in_db = index.ntotal

        vector_db = FAISS.from_embeddings(docs_embeddings_pairs, self.embedding_model, metadata)
        vector_db.save_local(self.vector_db_path)
        
        index = faiss.read_index(self.vector_db_path / "index.faiss")
        current_num_indexes_in_db = index.ntotal
        logger.info(f"Stored {current_num_indexes_in_db - num_indexes_in_db} new indexes in the vector db")

    def similarity_search(self, query, n_docs=3):
        if not self.flag_loaded_vector_db:
            self.vector_db = self.load_vector_db()
            self.flag_loaded_vector_db = True

        query_embedding = self.embedding_model.embed_documents([query])[0]
        relevant_docs = self.vector_db.similarity_search_by_vector(query_embedding, k=n_docs)

        return relevant_docs

