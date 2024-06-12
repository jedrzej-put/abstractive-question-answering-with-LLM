from pathlib import Path
import sys, os
from dotenv import load_dotenv

root_path = Path.cwd()
print(f"{root_path=}")
sys.path.append(str(root_path))
env_path = root_path / "src" / "config" / ".env"
print(f"Loaded env:{load_dotenv(dotenv_path=env_path)}")
print(f"{os.environ.get('HF_HOME')=}")



from src.config.configs import get_config
from src.lib.data_tools.ingest_data import load_data_to_vector_db

if __name__ == "__main__":
    config = get_config("config1")
    load_data_to_vector_db(config)