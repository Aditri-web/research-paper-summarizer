"""
evaluate.py
-----------
Compute ROUGE-L, BERTScore, and AlignScore on model predictions.

Usage:
    # Evaluate from a predictions JSONL file
    python src/evaluation/evaluate.py --config config/eval_config.yaml

    # Quick smoke test with synthetic data (no model needed)
    python src/evaluation/evaluate.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric computers
# ---------------------------------------------------------------------------


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(scores[k].fmeasure)

    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Compute BERTScore (Precision, Recall, F1).
    Uses DeBERTa-xlarge-mnli backbone for best correlation with human judgement.
    """
    import bert_score

    logger.info(f"  Computing BERTScore with {model_type} …")
    P, R, F1 = bert_score.score(
        predictions,
        references,
        model_type=model_type,
        lang="en",
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=True,
        verbose=False,
    )
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def compute_alignscore(
    predictions: list[str],
    sources: list[str],
    ckpt_path: str,
    evaluation_mode: str = "nli_sp",
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """
    Compute AlignScore (factual consistency: does prediction align with source?).
    Downloads checkpoint automatically on first run.

    AlignScore paper: https://arxiv.org/abs/2305.16842
    """
    try:
        from alignscore import AlignScore
    except ImportError:
        logger.error(
            "AlignScore not installed. Install it with:\n"
            "  pip install git+https://github.com/yuh-zha/AlignScore.git\n"
            "  python -m spacy download en_core_web_sm"
        )
        return {"alignscore": None, "alignscore_error": "AlignScore not installed"}

    # Auto-download checkpoint if not present
    if not os.path.isfile(ckpt_path):
        logger.info(f"AlignScore checkpoint not found at {ckpt_path} — downloading …")
        _download_alignscore_ckpt(ckpt_path)

    logger.info(f"  Computing AlignScore (mode={evaluation_mode}) …")
    scorer = AlignScore(
        model="roberta-large",
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode=evaluation_mode,
    )

    scores = scorer.score(contexts=sources, claims=predictions)
    mean_score = round(sum(scores) / len(scores), 4)
    return {"alignscore": mean_score}


def _download_alignscore_ckpt(ckpt_path: str) -> None:
    """Download the AlignScore-large checkpoint from HuggingFace."""
    import urllib.request

    ALIGNSCORE_URL = "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    logger.info(f"  Downloading AlignScore-large checkpoint to {ckpt_path} …")
    urllib.request.urlretrieve(ALIGNSCORE_URL, ckpt_path)
    logger.info("  ✅ AlignScore checkpoint downloaded.")


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def _write_markdown_report(results: dict, output_path: str) -> None:
    """Write a clean markdown evaluation report."""
    lines = [
        "# Evaluation Results — Mistral-7B Research Summarizer\n",
        f"**Total examples evaluated:** {results['num_examples']}\n",
        "## ROUGE Scores\n",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if k in results:
            lines.append(f"| {k} | {results[k]:.4f} |")

    lines += [
        "\n## BERTScore\n",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for k in ["bertscore_precision", "bertscore_recall", "bertscore_f1"]:
        if k in results:
            lines.append(f"| {k.replace('_', ' ').title()} | {results[k]:.4f} |")

    lines += ["\n## AlignScore (Factual Consistency)\n"]
    if results.get("alignscore") is not None:
        lines.append(f"| AlignScore | {results['alignscore']:.4f} |")
    else:
        lines.append("AlignScore: not computed (see logs).")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"📄 Markdown report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def evaluate(
    config: dict,
    predictions_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Load predictions JSONL and compute all configured metrics.

    Parameters
    ----------
    config : dict
        Parsed eval_config.yaml
    predictions_path : str, optional
        Override the predictions file path from config.
    dry_run : bool
        If True, use tiny synthetic data (no model needed).

    Returns
    -------
    dict with all metric scores.
    """
    preds_path = predictions_path or config["output"]["predictions_file"]

    if dry_run:
        logger.info("⚡ Dry-run mode — using synthetic data.")
        sources = [
            "We propose a novel transformer architecture for scientific summarization.",
            "This paper studies protein folding using deep learning methods.",
        ]
        references = [
            "A new transformer for scientific summarization.",
            "Deep learning applied to protein folding prediction.",
        ]
        predictions = [
            "Authors propose a transformer model for summarizing scientific papers.",
            "Study of protein folding via deep learning.",
        ]
    else:
        logger.info(f"Loading predictions from {preds_path} …")
        sources, references, predictions = [], [], []
        with open(preds_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                sources.append(item["source"])
                references.append(item["reference"])
                predictions.append(item["prediction"])

    logger.info(f"  Evaluating {len(predictions):,} examples …")
    results = {"num_examples": len(predictions)}

    # --- ROUGE ---
    if config["metrics"]["rouge"]["enabled"]:
        logger.info("📊 Computing ROUGE scores …")
        rouge_scores = compute_rouge(predictions, references)
        results.update(rouge_scores)
        logger.info(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")

    # --- BERTScore ---
    if config["metrics"]["bertscore"]["enabled"] and not dry_run:
        bsc = config["metrics"]["bertscore"]
        bertscore_scores = compute_bertscore(
            predictions,
            references,
            model_type=bsc["model_type"],
            batch_size=bsc["batch_size"],
            device=bsc.get("device", "cuda"),
        )
        results.update(bertscore_scores)
        logger.info(f"  BERTScore F1: {bertscore_scores['bertscore_f1']:.4f}")
    elif dry_run:
        # Lightweight BERTScore run for dry-run (use small model)
        logger.info("📊 Computing BERTScore (dry-run, small model) …")
        bertscore_scores = compute_bertscore(
            predictions,
            references,
            model_type="distilbert-base-uncased",
            batch_size=2,
            device="cpu",
        )
        results.update(bertscore_scores)

    # --- AlignScore ---
    if config["metrics"]["alignscore"]["enabled"] and not dry_run:
        asc = config["metrics"]["alignscore"]
        alignscore_scores = compute_alignscore(
            predictions=predictions,
            sources=sources,
            ckpt_path=asc["ckpt_path"],
            evaluation_mode=asc["evaluation_mode"],
            batch_size=asc["batch_size"],
            device=asc.get("device", "cuda"),
        )
        results.update(alignscore_scores)
        if results.get("alignscore"):
            logger.info(f"  AlignScore: {results['alignscore']:.4f}")
    elif dry_run:
        results["alignscore"] = None
        results["alignscore_note"] = "Skipped in dry-run mode"

    # --- Save results ---
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = config["output"]["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Metrics saved to {metrics_path}")

    # --- Markdown report ---
    report_path = config["output"]["report_file"]
    _write_markdown_report(results, report_path)

    # --- Print summary table ---
    print("\n" + "=" * 50)
    print("  EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in results.items():
        if k == "num_examples":
            continue
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<30} {val_str}")
    print("=" * 50 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate summarizer with ROUGE-L, BERTScore, AlignScore"
    )
    parser.add_argument("--config", default="config/eval_config.yaml")
    parser.add_argument("--predictions", default=None, help="Path to predictions JSONL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic data (no model needed)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate(config, predictions_path=args.predictions, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
