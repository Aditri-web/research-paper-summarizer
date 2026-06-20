"""
summarize.py
------------
CLI inference tool for the fine-tuned Mistral-7B summarizer.

Usage:
    # From stdin
    python src/inference/summarize.py --text "Your paper abstract here..."

    # From a file
    python src/inference/summarize.py --input paper.txt

    # With custom adapter path
    python src/inference/summarize.py --input paper.txt \
        --adapter models/mistral7b-summarizer-qlora
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(
    adapter_path: str = "models/mistral7b-summarizer-qlora",
    load_in_4bit: bool = True,
):
    """
    Load model + tokenizer. Prefers Unsloth (GPU), falls back to PEFT.
    """
    try:
        from unsloth import FastLanguageModel

        logger.info(f"Loading model via Unsloth from {adapter_path} …")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        logger.info("✅ Model loaded (Unsloth fast inference).")
        return model, tokenizer

    except ImportError:
        logger.warning("Unsloth not available — using standard PEFT.")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_name = "mistralai/Mistral-7B-Instruct-v0.3"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Model loaded (PEFT).")
        return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def summarize(
    text: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate a summary for the given paper text.

    Parameters
    ----------
    text : str
        Paper abstract or full text (will be truncated to model max_length).

    Returns
    -------
    str : Generated summary.
    """
    import torch

    from src.data.preprocess import INFERENCE_TEMPLATE

    prompt = INFERENCE_TEMPLATE.format(source=text.strip())
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return decoded.removeprefix("Summary:").strip()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a research paper with the fine-tuned Mistral-7B model."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", type=str, help="Paper text as a string.")
    source_group.add_argument("--input", type=str, help="Path to a .txt file with the paper.")

    parser.add_argument(
        "--adapter",
        type=str,
        default="models/mistral7b-summarizer-qlora",
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    # Read source text
    if args.input:
        with open(args.input, "r") as f:
            text = f.read()
    else:
        text = args.text

    if not text.strip():
        logger.error("Input text is empty.")
        sys.exit(1)

    # Load & run
    model, tokenizer = load_model(
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
    )

    summary = summarize(
        text=text,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\n" + "=" * 60)
    print("GENERATED SUMMARY")
    print("=" * 60)
    print(summary)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
