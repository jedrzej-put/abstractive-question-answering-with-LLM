import torch
import logging

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


def sort_docs_by_len(id_title_list, text_list, len_list) -> tuple[list[dict[str, str]], list[str], list[int]]:
    logging.info(f"Sorting the documents based on the length of the text")

    combined_list = list(zip(id_title_list, text_list, len_list))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[2])
    id_title_list, text_list, len_list = zip(*sorted_combined_list)

    return id_title_list, text_list, len_list