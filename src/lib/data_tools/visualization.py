import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
from transformers import AutoTokenizer
from src.config.Config import Config

def hist_value_counts(data: pd.Series, title: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None, save_path: Optional[str] = None, translation_map: Optional[dict[str, str]] = None):
    plt.figure(figsize=(12, 8))

    if translation_map:
        data = data.map(translation_map).fillna(data)

    value_counts = data.value_counts().sort_index()

    # Create the bar plot from the value counts
    ax = value_counts.plot(kind="bar", edgecolor="black")

    # Set x-axis to show only integer values
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=90)

    for i, value in enumerate(value_counts):
        ax.text(i, value, str(value), ha="center", va="bottom")
    # Display the plot
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
    plt.title(title)
    plt.show()

def count_tokens(df: pd.DataFrame, config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model_name)
    df.loc[:, "question_token_count"] = df["question"].progress_apply(lambda x: len(tokenizer.tokenize(x)))
    df.loc[:, "passage_text_token_count"] = df["passage_text"].progress_apply(lambda x: len(tokenizer.tokenize(x)))

