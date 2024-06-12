from pathlib import Path
import sys

root_path = Path.cwd()
print(f"{root_path=}")
sys.path.append(str(root_path))

from src.config.configs import get_config
from src.lib.data_tools.ingest_data import load_data_to_vector_db



if __name__ == "__main__":
    config = get_config("config1")
    load_data_to_vector_db(config)