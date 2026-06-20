# Research Paper Summarizer 📄🤖

> Fine-tune **Mistral-7B-Instruct-v0.3** with **QLoRA** (via Unsloth) to generate concise, accurate summaries of scientific research papers.

[![Lint](https://github.com/YOUR_USERNAME/research-paper-summarizer/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/research-paper-summarizer/actions/workflows/lint.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/model-Mistral--7B--Instruct--v0.3-blueviolet)
![Method](https://img.shields.io/badge/fine--tuning-QLoRA-orange)
![Framework](https://img.shields.io/badge/framework-Unsloth-green)

---

## Overview

This project fine-tunes Mistral-7B-Instruct using **QLoRA** (4-bit quantised Low-Rank Adaptation) for scientific paper summarization. It combines two complementary datasets — **ArXiv** (long structured papers) and **SciTLDR** (short CS paper TLDRs) — and evaluates with three metrics that cover lexical overlap, semantic fidelity, and factual consistency.

## Architecture

```
Data Pipeline
  ├── ccdv/arxiv-summarization     (10k train / 1k val / 1k test)
  └── allenai/scitldr (AIC split)  (full ~3k samples)
        ↓
  Merge + Mistral [INST] prompt formatting
        ↓
Training (Google Colab / Kaggle GPU)
  ├── Base: mistralai/Mistral-7B-Instruct-v0.3
  ├── QLoRA: 4-bit NF4, r=16, α=32
  ├── Framework: Unsloth + TRL SFTTrainer
  └── Optimizer: AdamW 8-bit, cosine LR, 3 epochs
        ↓
Evaluation
  ├── ROUGE-L         (lexical overlap)
  ├── BERTScore       (semantic similarity, DeBERTa-xlarge)
  └── AlignScore      (factual consistency, NLI-based)
        ↓
Deployment
  └── Gradio Web App  (local / HuggingFace Spaces)
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Base Model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Fine-tuning | QLoRA (4-bit NF4) |
| Training Framework | [Unsloth](https://github.com/unslothai/unsloth) + TRL SFTTrainer |
| Datasets | ArXiv Summarization + SciTLDR |
| ROUGE | `rouge-score` |
| BERTScore | `bert-score` (DeBERTa-xlarge-mnli) |
| AlignScore | [AlignScore](https://github.com/yuh-zha/AlignScore) |
| Demo | Gradio |

## Project Structure

```
research-paper-summarizer/
├── config/
│   ├── training_config.yaml    # All hyperparameters
│   └── eval_config.yaml        # Evaluation settings
├── src/
│   ├── data/
│   │   ├── dataset_loader.py   # Load & merge ArXiv + SciTLDR
│   │   └── preprocess.py       # Mistral prompt formatting
│   ├── training/
│   │   └── train.py            # QLoRA training script
│   ├── evaluation/
│   │   ├── evaluate.py         # ROUGE-L + BERTScore + AlignScore
│   │   └── generate_summaries.py  # Batch inference
│   └── inference/
│       └── summarize.py        # CLI single-paper inference
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_training.ipynb       # ← Run on Colab/Kaggle
│   ├── 02_evaluation.ipynb
│   └── 03_inference_demo.ipynb
├── deployment/
│   └── app.py                  # Gradio web app
├── docs/
├── .github/workflows/lint.yml
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/research-paper-summarizer.git
cd research-paper-summarizer

# Dev environment (no CUDA needed for EDA + evaluation)
pip install -r requirements-dev.txt
pip install datasets transformers pyyaml rouge-score
```

### 2. Explore Data (CPU-safe)

```bash
jupyter notebook notebooks/00_data_exploration.ipynb
```

### 3. Train (GPU required — Colab / Kaggle)

Open `notebooks/01_training.ipynb` on **Google Colab (A100)** or **Kaggle (T4/P100)**.

Or run the script directly on a GPU machine:

```bash
# Install Unsloth first
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt

python src/training/train.py --config config/training_config.yaml
```

### 4. Generate Predictions

```bash
python src/evaluation/generate_summaries.py --config config/eval_config.yaml
```

### 5. Evaluate

```bash
# Full evaluation (requires GPU for BERTScore + AlignScore)
python src/evaluation/evaluate.py --config config/eval_config.yaml

# Smoke test with synthetic data (CPU, no model needed)
python src/evaluation/evaluate.py --config config/eval_config.yaml --dry-run
```

### 6. Run Gradio Demo

```bash
python deployment/app.py
# Open http://localhost:7860
```

### 7. Summarize a Single Paper

```bash
python src/inference/summarize.py --input path/to/paper.txt
# or
python src/inference/summarize.py --text "Paste your abstract here..."
```

## Evaluation Results

*After training, fill in your results below:*

| Metric | Score |
|--------|-------|
| ROUGE-1 | — |
| ROUGE-2 | — |
| **ROUGE-L** | — |
| BERTScore F1 | — |
| **AlignScore** | — |

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size (effective) | 8 (2 × grad_accum 4) |
| Epochs | 3 |
| Max sequence length | 2048 |
| Quantization | 4-bit NF4 |
| Optimizer | AdamW 8-bit |
| Scheduler | Cosine |

## AlignScore Setup

AlignScore requires an additional checkpoint (~500 MB) and spaCy model:

```bash
pip install git+https://github.com/yuh-zha/AlignScore.git
python -m spacy download en_core_web_sm
# Checkpoint is auto-downloaded to alignscore_ckpt/ on first run
```

## License

MIT License — see [LICENSE](LICENSE)

## Citation

```bibtex
@misc{research-paper-summarizer-2024,
  title  = {Research Paper Summarizer: Mistral-7B QLoRA Fine-tuning},
  author = {Your Name},
  year   = {2024},
  url    = {https://github.com/YOUR_USERNAME/research-paper-summarizer}
}
```
