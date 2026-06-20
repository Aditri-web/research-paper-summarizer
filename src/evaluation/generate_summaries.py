"""
generate_summaries.py
---------------------
Batch inference script: loads the trained adapter + base model and generates
summaries for the test split, saving results to a JSONL file.

Usage:
    python src/evaluation/generate_summaries.py \
        --config config/eval_config.yaml \
        --output results/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: dict):
    """
    Load the fine-tuned model.
    Tries Unsloth first (GPU), falls back to plain PEFT (CPU/MPS).
    """
    adapter_path = config["model"]["adapter_path"]
    base_model = config["model"]["base_model"]
    load_in_4bit = config["model"].get("load_in_4bit", True)

    try:
        from unsloth import FastLanguageModel

        logger.info("Loading model via Unsloth …")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    except ImportError:
        logger.warning("Unsloth not available — falling back to PEFT + transformers.")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], config: dict) -> list[str]:
    """Generate summaries for a list of prompt strings."""
    import torch

    gen_cfg = config["model"]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            temperature=gen_cfg.get("temperature", 0.1),
            do_sample=gen_cfg.get("do_sample", False),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    summaries = []
    for out in outputs:
        decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
        # Strip "Summary:" prefix if model echoed it
        decoded = decoded.removeprefix("Summary:").strip()
        summaries.append(decoded)
    return summaries


def run_inference(config: dict, output_path: Optional[str] = None) -> list[dict]:
    """
    Run batch inference on the test split.

    Returns
    -------
    List of dicts with keys: source, reference, prediction
    """
    from datasets import load_from_disk

    from src.data.preprocess import INFERENCE_TEMPLATE

    output_path = output_path or config["output"]["predictions_file"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Load test dataset ---
    processed_dir = config["data"]["processed_dir"]
    logger.info(f"Loading test data from {processed_dir} …")
    ds = load_from_disk(processed_dir)
    test_ds = ds[config["data"]["test_split"]]

    max_samples = config["data"].get("max_eval_samples")
    if max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))

    logger.info(f"Evaluating on {len(test_ds):,} test examples …")

    # The processed dataset only has 'text' column (full prompt+answer).
    # We need to re-load raw test data to get source/reference separately.
    # Re-load raw to separate source & target.
    # Load raw combined test
    raw_processed_dir = processed_dir + "_raw"
    if os.path.isdir(raw_processed_dir):
        from datasets import load_from_disk as _lfd

        raw_test = _lfd(raw_processed_dir)["test"]
    else:
        logger.info("Raw test split not found — rebuilding from HuggingFace Hub for references.")
        from src.data.dataset_loader import load_config, load_datasets

        train_cfg = load_config()
        _, _, raw_test = load_datasets(train_cfg)

    if max_samples:
        raw_test = raw_test.select(range(min(max_samples, len(raw_test))))

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(config)

    # --- Batch inference ---
    batch_size = config["data"].get("batch_size", 8)
    results = []
    for i in tqdm(range(0, len(raw_test), batch_size), desc="Generating"):
        batch = raw_test[i : i + batch_size]
        sources = batch["source"]
        references = batch["target"]

        prompts = [INFERENCE_TEMPLATE.format(source=s.strip()) for s in sources]
        predictions = generate_batch(model, tokenizer, prompts, config)

        for src, ref, pred in zip(sources, references, predictions):
            results.append({"source": src, "reference": ref, "prediction": pred})

    # --- Save ---
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"✅ Predictions saved to {output_path}  ({len(results):,} examples)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch inference for summarizer")
    parser.add_argument("--config", default="config/eval_config.yaml")
    parser.add_argument("--output", default=None, help="Override output JSONL path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_inference(config, output_path=args.output)


if __name__ == "__main__":
    main()
