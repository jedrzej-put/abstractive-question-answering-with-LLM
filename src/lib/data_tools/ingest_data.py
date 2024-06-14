import glob
import json
import logging
from functools import lru_cache

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from src.lib.data_tools.read_file import process_jsonl
from src.config.Config import Config
from src.lib.common.tools import sort_docs_by_len, get_overwrite_console_logger
from src.lib.VectorDb.tools import get_vector_db
from src.lib.Embeddings.tools import get_embedding_model

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()
overwrite_console_logger = get_overwrite_console_logger()


def split_documents_into_chunks(config, texts, metadatas) -> tuple[tuple[str], tuple[dict]]:
    overwrite_console_logger.info("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=config.max_chunk_size, chunk_overlap=config.chunk_overlap, separator=config.text_splitter_separator
    )
    texts_chunks = text_splitter.create_documents(texts, metadatas)
    texts = [text.page_content for text in texts_chunks]
    metadatas = [text.metadata for text in texts_chunks]

    texts, metadatas = sort_docs_by_len(texts, metadatas)
    
    return texts, metadatas

def load_data_to_vector_db(config: Config):
    logger.info(f"Loading embeddings model...")
    embedding_model = get_embedding_model(config)
    logger.info("Embeddings model loaded successfully")

    vector_db = get_vector_db(config)
    logger.info("Vector database loaded successfully")

    text_counter = 0
    for texts, metadatas in process_jsonl(config):
        # if config.split_into_chunks:
        #     texts, metadatas = split_documents_into_chunks(config, texts, metadatas)
        
        logger.debug(f"Calculating embeddings for {len(texts)} documents...")
        embeddings = embedding_model.embed_documents(texts)
        texts_embeddings_pairs = list(zip(texts, embeddings))
        
        logger.debug(f"Storing embeddings in vector database...")
        vector_db.store_embeddings(texts_embeddings_pairs, metadatas, logger=overwrite_console_logger)
        text_counter += len(texts)
        logger.debug(f"Number of documents processed so far: {text_counter}")

    logger.info(f"Finished loading data to vector database")
    logger.info("Total number of documents processed: {text_counter}")



