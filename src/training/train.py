"""
train.py
--------
QLoRA fine-tuning of Mistral-7B-Instruct-v0.3 using Unsloth + TRL SFTTrainer.

Run from project root (on Colab/Kaggle GPU):
    python src/training/train.py --config config/training_config.yaml

Or import and call `main(config)` from the training notebook.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guard: Unsloth must be installed on a CUDA machine
# ---------------------------------------------------------------------------
def _check_unsloth():
    try:
        import unsloth  # noqa: F401
    except ImportError:
        logger.error(
            "Unsloth is not installed. Please install it first:\n"
            "  pip install unsloth\n"
            "  # or follow https://github.com/unslothai/unsloth#installation"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(config: dict) -> None:
    """
    Full QLoRA training pipeline.

    Parameters
    ----------
    config : dict
        Parsed training_config.yaml
    """
    _check_unsloth()

    # --- Imports (deferred so import errors are clear) ---
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    # --- 1. Load or build processed dataset ---
    processed_dir = config["data"]["processed_dir"]
    if os.path.isdir(processed_dir):
        logger.info(f"Loading processed dataset from {processed_dir} …")
        from datasets import load_from_disk as _lfd

        processed = _lfd(processed_dir)
        train_ds = processed["train"]
        val_ds = processed["validation"]
    else:
        logger.info("Processed dataset not found — building from scratch …")
        from src.data.dataset_loader import load_datasets
        from src.data.preprocess import prepare_and_save

        raw_train, raw_val, _ = load_datasets(config)
        # We need a tokenizer to format; load it first temporarily
        from transformers import AutoTokenizer

        tokenizer_tmp = AutoTokenizer.from_pretrained(
            config["model"]["name"], trust_remote_code=True
        )
        if tokenizer_tmp.pad_token is None:
            tokenizer_tmp.pad_token = tokenizer_tmp.eos_token
        processed = prepare_and_save(raw_train, raw_val, raw_val, tokenizer_tmp, config)
        train_ds = processed["train"]
        val_ds = processed["validation"]

    # --- 2. Load base model + tokenizer (Unsloth 4-bit) ---
    logger.info(f"Loading base model: {config['model']['name']} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=None,  # auto-detect
        load_in_4bit=config["model"]["load_in_4bit"],
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 3. Apply QLoRA adapters ---
    logger.info("Applying QLoRA adapters …")
    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=config["data"]["seed"],
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # --- 4. Training arguments ---
    tc = config["training"]
    training_args = TrainingArguments(
        output_dir=tc["output_dir"],
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        warmup_steps=tc["warmup_steps"],
        learning_rate=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        optim=tc["optim"],
        fp16=tc.get("fp16", False),
        bf16=tc.get("bf16", True),
        logging_steps=tc["logging_steps"],
        eval_steps=tc["eval_steps"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        greater_is_better=tc["greater_is_better"],
        evaluation_strategy="steps",
        report_to=tc.get("report_to", "none"),
        seed=config["data"]["seed"],
    )

    # --- 5. SFTTrainer ---
    logger.info("Initialising SFTTrainer …")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field=tc["dataset_text_field"],
        max_seq_length=config["model"]["max_seq_length"],
        packing=tc["packing"],
        args=training_args,
    )

    # --- 6. Train ---
    logger.info("🚀 Starting training …")
    trainer_stats = trainer.train()
    logger.info(
        f"✅ Training complete! "
        f"Steps: {trainer_stats.global_step} | "
        f"Loss: {trainer_stats.training_loss:.4f}"
    )

    # --- 7. Save adapter ---
    output_dir = tc["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"💾 Adapter saved to {output_dir}")

    # --- 8. Optional: push to Hub ---
    hub_cfg = config.get("hub", {})
    if hub_cfg.get("push_to_hub") and hub_cfg.get("hub_model_id"):
        hf_token = os.environ.get(hub_cfg.get("hub_token_env", "HF_TOKEN"))
        if hf_token:
            logger.info(f"Pushing model to HuggingFace Hub: {hub_cfg['hub_model_id']} …")
            model.push_to_hub(hub_cfg["hub_model_id"], token=hf_token)
            tokenizer.push_to_hub(hub_cfg["hub_model_id"], token=hf_token)
            logger.info("✅ Model pushed to Hub successfully.")
        else:
            logger.warning("HF_TOKEN env var not set — skipping Hub push.")

    return trainer_stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
