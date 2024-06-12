from pathlib import Path
from src.lib.VectorDb.FAISSVectorDb import FAISSVectorDb
from src.lib.Embeddings.HFEmbeddings import HFMultilingualE5Embeddings
from src.config.Config import Config

root_path = Path.cwd()
vector_db_paths = root_path / "data" / "vector_databases"
def get_config(config_name: str) -> Config:
    if config_name not in configs:
        raise ValueError(f"Config {config_name} not found")
    return configs[config_name]

configs = {
    "config1": Config(
        passages_path = root_path / "data" / "ipipan-polqa"/ "passages.jsonl",
        vector_db=FAISSVectorDb,
        vector_db_path=vector_db_paths / "faiss_vector_db_config_1",
        embedding_model=HFMultilingualE5Embeddings,
        max_chunk_size=512,
        chunk_overlap=200,
        split_into_chunks=True,
        text_splitter_separator="",
    ),
}


