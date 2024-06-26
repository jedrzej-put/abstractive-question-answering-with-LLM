import json
from typing import Generator, Dict, Any
from pathlib import Path
from tqdm import tqdm
from src.lib.common.tools import sort_docs_by_len
from src.config.Config import Config
from src.config.configs import get_config
from src.lib.common.tools import logger

# Generator function to read and process a JSONL file line by line
def read_jsonl_file(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON object from each line
            json_obj = json.loads(line.strip())
            yield json_obj


def process_jsonl(config: Config, batch_size: int=1024) -> Generator[tuple[tuple[str], tuple[dict[str, str]]], None, None]:
    logger.info(f"Starting to process file: {config.passages_path}")
    metadatas, texts = [], []
    
    num_lines = sum(1 for _ in open(config.passages_path, 'r', encoding='utf-8'))
    logger.info(f"Total number of lines in the file: {num_lines}")
    
    for i, json_obj in enumerate(tqdm(read_jsonl_file(config.passages_path), total=num_lines, desc="Processing JSONL"), start=1):
        if 'id' in json_obj and 'title' in json_obj and 'text' in json_obj:
            metadatas.append({"id": json_obj['id'], "title": json_obj['title']})
            texts.append(json_obj['text'])
        if i % batch_size == 0:
            texts, metadatas = sort_docs_by_len(texts, metadatas)
            yield texts, metadatas
            texts, metadatas = [], []

    if metadatas and texts:
        texts, metadatas = sort_docs_by_len(texts, metadatas)
        yield texts, metadatas

    logger.info(f"Finished processing file: {config.passages_path}")

if __name__ == "__main__":
    config = get_config("config1")
    metadatas, texts = process_jsonl(config)
    print(f"Number of documents: {len(texts)}")
    print(f"{metadatas[:3]=}")
    print(f"{texts[:3]=}")