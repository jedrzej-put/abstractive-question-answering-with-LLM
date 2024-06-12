import glob
import json
import logging
from functools import lru_cache

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from src.lib.data_tools.read_file import process_jsonl
from src.config.Config import Config
from src.lib.common.tools import sort_docs_by_len
from src.lib.VectorDb.tools import get_vector_db
from src.lib.Embeddings.tools import get_embedding_model

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


def load_data_to_vector_db(config: Config):
    metadata, docs, lens = process_jsonl(config)
    logger.info(f"Number of documents: {len(docs)}")
    
    if config.split_into_chunks:
        logger.info("Splitting documents into chunks...")
        text_splitter = CharacterTextSplitter(
            chunk_size=config.max_chunk_size, chunk_overlap=config.chunk_overlap, separator=config.text_splitter_separator
        )
        docs_chunks = text_splitter.create_documents(docs, metadata)
        docs = [doc.page_content for doc in docs_chunks]
        metadata = [doc.metadata for doc in docs_chunks]
        lens = [len(doc) for doc in docs]

        metadata, docs, lens = sort_docs_by_len(metadata, docs, lens)

    logger.info(f"Calculating embeddings for {len(docs)} documents...")
    embedding_model = get_embedding_model(config)
    embeddings = embedding_model.embed_documents(docs)
    docs_embeddings_pairs = list(zip(docs, embeddings))

    logger.info(f"Storing embeddings in vector database...")
    vector_db = get_vector_db(config)
    vector_db.store_embeddings(docs_embeddings_pairs, metadata)



