"""
preprocess.py
-------------
Tokenization and prompt-formatting utilities for the summarizer.

Usage:
    from src.data.preprocess import build_prompt, tokenize_dataset
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "[INST] You are a scientific paper summarizer. "
    "Write a concise, accurate summary of the following research paper.\n\n"
    "Paper:\n{source}\n\n[/INST] Summary: {target}"
)

INFERENCE_TEMPLATE = (
    "[INST] You are a scientific paper summarizer. "
    "Write a concise, accurate summary of the following research paper.\n\n"
    "Paper:\n{source}\n\n[/INST] Summary:"
)


def build_prompt(source: str, target: str = "", inference: bool = False) -> str:
    """
    Format a source/target pair into a Mistral instruction prompt.

    Parameters
    ----------
    source : str
        The paper text / abstract.
    target : str
        The reference summary (empty during inference).
    inference : bool
        If True, return prompt without the target answer (for generation).
    """
    if inference:
        return INFERENCE_TEMPLATE.format(source=source.strip())
    return PROMPT_TEMPLATE.format(source=source.strip(), target=target.strip())


# ---------------------------------------------------------------------------
# Dataset tokenization
# ---------------------------------------------------------------------------


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    config: dict,
    split_name: str = "train",
) -> Dataset:
    """
    Apply prompt template and tokenize a dataset split.

    Parameters
    ----------
    dataset : datasets.Dataset
        Must have 'source' and 'target' columns.
    tokenizer : PreTrainedTokenizer
        Loaded tokenizer (with pad token set).
    config : dict
        Parsed training_config.yaml.
    split_name : str
        Name for logging.

    Returns
    -------
    datasets.Dataset with a 'text' column (formatted prompt strings).
    """

    def format_example(batch):
        texts = []
        for src, tgt in zip(batch["source"], batch["target"]):
            texts.append(build_prompt(src, tgt))
        return {"text": texts}

    logger.info(f"  Formatting prompts for {split_name} split …")
    dataset = dataset.map(format_example, batched=True, remove_columns=["source", "target"])

    logger.info(f"  ✅ {split_name}: {len(dataset):,} examples formatted.")
    return dataset


def prepare_and_save(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    tokenizer,
    config: dict,
    output_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Format all splits and optionally save to disk.

    Returns
    -------
    DatasetDict with keys 'train', 'validation', 'test'.
    """
    output_dir = output_dir or config["data"].get("processed_dir", "data/processed")

    processed = DatasetDict(
        {
            "train": tokenize_dataset(train_ds, tokenizer, config, "train"),
            "validation": tokenize_dataset(val_ds, tokenizer, config, "validation"),
            "test": tokenize_dataset(test_ds, tokenizer, config, "test"),
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    processed.save_to_disk(output_dir)
    logger.info(f"💾 Processed dataset saved to {output_dir}")
    return processed
