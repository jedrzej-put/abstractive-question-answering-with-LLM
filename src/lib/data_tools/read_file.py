import json
from typing import Generator, Dict, Any
from pathlib import Path
from tqdm import tqdm
import logging 
from src.lib.common.tools import sort_docs_by_len
from src.config.Config import Config
from src.config.configs import get_config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generator function to read and process a JSONL file line by line
def read_jsonl_file(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON object from each line
            json_obj = json.loads(line.strip())
            yield json_obj


def process_jsonl(config) -> tuple[tuple[dict[str, str]], tuple[str], tuple[int]]:
    
    logging.info(f"Starting to process file: {config.passages_path}")
    id_title_list = []
    text_list = []
    len_list = []
    
    num_lines = sum(1 for _ in open(config.passages_path, 'r', encoding='utf-8'))
    logging.info(f"Total number of lines in the file: {num_lines}")
    for json_obj in tqdm(read_jsonl_file(config.passages_path), total=num_lines, desc="Processing JSONL"):
        if 'id' in json_obj and 'title' in json_obj and 'text' in json_obj:
            id_title_list.append({"id": json_obj['id'], "title": json_obj['title']})
            text_list.append(json_obj['text'])
            len_list.append(len(json_obj['text']))

    id_title_list, text_list, len_list = sort_docs_by_len(id_title_list, text_list, len_list)
    
    logging.info(f"Number of id-title pairs processed: {len(id_title_list)}")
    logging.info(f"Number of texts processed: {len(text_list)}")
    
    return id_title_list, text_list, len_list

if __name__ == "__main__":
    config = get_config("config1")
    id_title_list, text_list = process_jsonl(config)
    print(f"Number of documents: {len(text_list)}")
    print(f"{id_title_list[:3]=}")
    print(f"{text_list[:3]=}")