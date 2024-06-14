import torch
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_device():
    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logger.info(f"device: {device}")
    return device


def sort_docs_by_len(texts, metadatas) -> tuple[list[str], list[dict[str, str]]]:
    logging.info(f"Sorting the documents based on the length of the text")

    combined_list = list(zip(texts, metadatas))
    sorted_combined_list = sorted(combined_list, key=lambda x: len(x[0]))
    texts, metadatas = zip(*sorted_combined_list)

    return texts, metadatas



class OverwriteConsoleHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=sys.stdout)

    def emit(self, record):
        message = self.format(record)
        sys.stdout.write(f"\r{message}{' ' * (180 - len(message))}")
        sys.stdout.flush()

def get_overwrite_console_logger() -> logging.Logger:
    logger = logging.getLogger('overwriteConsoleLogger')
    logger.setLevel(logging.INFO)
    handler = OverwriteConsoleHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
