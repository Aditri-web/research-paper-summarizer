"""
dataset_loader.py
-----------------
Loads and merges ArXiv + SciTLDR datasets from HuggingFace Hub.

Usage:
    from src.data.dataset_loader import load_datasets
    train_ds, val_ds, test_ds = load_datasets(config)
"""

from __future__ import annotations

import logging
from typing import Optional

import yaml
from datasets import concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_split(dataset_split, n: Optional[int], seed: int):
    """Return at most n examples from a split, shuffled."""
    if n is None or n >= len(dataset_split):
        return dataset_split.shuffle(seed=seed)
    return dataset_split.shuffle(seed=seed).select(range(n))


def _load_arxiv(config: dict, seed: int):
    """
    Load ccdv/arxiv-summarization.
    Columns: article, abstract  →  renamed to source, target
    """
    logger.info("📥 Loading ArXiv summarization dataset …")
    ds = load_dataset(config["data"]["arxiv_dataset"], trust_remote_code=True)

    # Rename columns to unified schema
    ds = ds.rename_columns({"article": "source", "abstract": "target"})

    # Keep only text columns
    cols_to_keep = ["source", "target"]
    ds = ds.select_columns(cols_to_keep)

    train = _sample_split(ds["train"], config["data"]["arxiv_train_size"], seed)
    val = _sample_split(ds["validation"], config["data"]["arxiv_val_size"], seed)
    test = _sample_split(ds["test"], config["data"]["arxiv_test_size"], seed)

    logger.info(f"  ArXiv → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


def _load_scitldr(config: dict, seed: int):
    """
    Load allenai/scitldr (AIC split).
    Columns: source (list of sentences), target (list of TLDRs)
    We join source sentences and take the first target TLDR.
    """
    logger.info("📥 Loading SciTLDR dataset …")
    cfg_name = config["data"]["scitldr_config"]
    ds = load_dataset(config["data"]["scitldr_dataset"], cfg_name, trust_remote_code=True)

    def flatten(batch):
        """Join source sentence list; pick first TLDR as target."""
        sources, targets = [], []
        for src, tgt in zip(batch["source"], batch["target"]):
            # source is a list of strings; join with space
            sources.append(" ".join(src) if isinstance(src, list) else src)
            # target is a list; take the first TLDR
            targets.append(tgt[0] if isinstance(tgt, list) else tgt)
        return {"source": sources, "target": targets}

    ds = ds.map(flatten, batched=True, remove_columns=ds["train"].column_names)

    # Remove very short examples (likely parsing errors)
    ds = ds.filter(lambda x: len(x["source"]) > 100 and len(x["target"]) > 10)

    train = ds["train"].shuffle(seed=seed)
    val = ds["validation"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=seed)

    logger.info(f"  SciTLDR → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_datasets(config: dict) -> tuple:
    """
    Load, merge, and return train / val / test splits.

    Parameters
    ----------
    config : dict
        Parsed training_config.yaml (or equivalent dict).

    Returns
    -------
    train_ds, val_ds, test_ds : datasets.Dataset
    """
    seed = config["data"].get("seed", 42)

    arxiv_train, arxiv_val, arxiv_test = _load_arxiv(config, seed)
    scitldr_train, scitldr_val, scitldr_test = _load_scitldr(config, seed)

    train_ds = concatenate_datasets([arxiv_train, scitldr_train]).shuffle(seed=seed)
    val_ds = concatenate_datasets([arxiv_val, scitldr_val]).shuffle(seed=seed)
    test_ds = concatenate_datasets([arxiv_test, scitldr_test]).shuffle(seed=seed)

    logger.info(
        f"✅ Combined → train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}"
    )
    return train_ds, val_ds, test_ds


def load_config(config_path: str = "config/training_config.yaml") -> dict:
    """Utility to load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    train, val, test = load_datasets(cfg)
    print(f"\nSample train example:\n  source[:200]: {train[0]['source'][:200]}")
    print(f"  target: {train[0]['target']}")
