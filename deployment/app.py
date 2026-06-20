"""
app.py — Gradio Web Demo
------------------------
Research Paper Summarizer powered by Mistral-7B + QLoRA.

Run locally:
    python deployment/app.py

Deploy to HuggingFace Spaces:
    Upload this file + requirements.txt to a Gradio Space.
    Set adapter_path to your HF model repo ID.
"""

from __future__ import annotations

import logging
import os
import sys

import gradio as gr

# Add project root to path when running from deployment/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "models/mistral7b-summarizer-qlora")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

EXAMPLE_ABSTRACTS = [
    (
        "ArXiv: Attention Is All You Need",
        (
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks that include an encoder and a decoder. The best "
            "performing models also connect the encoder and decoder through an attention mechanism. "
            "We propose a new simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments "
            "on two machine translation tasks show these models to be superior in quality while being "
            "more parallelizable and requiring significantly less time to train. Our model achieves "
            "28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the "
            "existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 "
            "English-to-French translation task, our model establishes a new single-model "
            "state-of-the-art BLEU score of 41.0."
        ),
    ),
    (
        "ArXiv: LoRA Low-Rank Adaptation",
        (
            "An important paradigm of natural language processing consists of large-scale pre-training "
            "on general domain data and adaptation to particular tasks or domains. As we pre-train "
            "larger models, full fine-tuning, which retrains all model parameters, becomes less "
            "feasible. Using GPT-3 175B as an example -- deploying independent instances of "
            "fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose "
            "Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects "
            "trainable rank decomposition matrices into each layer of the Transformer architecture, "
            "greatly reducing the number of trainable parameters for downstream tasks. Compared to "
            "GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters "
            "by 10,000 times and the GPU memory requirement by 3 times."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Model (lazy-loaded on first call)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading model (first call) …")
        try:
            from unsloth import FastLanguageModel

            _model, _tokenizer = FastLanguageModel.from_pretrained(
                model_name=ADAPTER_PATH,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=LOAD_IN_4BIT,
            )
            FastLanguageModel.for_inference(_model)
        except (ImportError, Exception) as e:
            logger.warning(f"Unsloth load failed ({e}) — using PEFT fallback.")
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb = BitsAndBytesConfig(load_in_4bit=True) if LOAD_IN_4BIT else None
            base = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
            _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            _model.eval()
            _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------


def generate_summary(text: str, max_tokens: int, temperature: float) -> tuple[str, str]:
    """
    Generate summary and compute ROUGE-L vs. (optional) reference.
    Returns (summary, status_message).
    """
    if not text.strip():
        return "", "⚠️ Please enter some paper text."

    import torch

    from src.data.preprocess import INFERENCE_TEMPLATE

    prompt = INFERENCE_TEMPLATE.format(source=text.strip())

    try:
        model, tokenizer = _get_model()
    except Exception as e:
        return "", f"❌ Model load error: {e}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=temperature > 0.01,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    summary = decoded.removeprefix("Summary:").strip()

    # Word count info
    word_count = len(summary.split())
    compression = len(text.split()) / max(word_count, 1)
    status = f"✅ Generated {word_count} words  |  Compression ratio: {compression:.1f}×"

    return summary, status


def evaluate_against_reference(summary: str, reference: str) -> str:
    """Compute ROUGE-L between generated summary and user-provided reference."""
    if not summary.strip() or not reference.strip():
        return "Provide both a generated summary and a reference to compute ROUGE-L."

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference.strip(), summary.strip())
    rl = score["rougeL"].fmeasure
    return f"📊 ROUGE-L F1: **{rl:.4f}**"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1100px;
}
.title-block {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="Research Paper Summarizer — Mistral-7B QLoRA",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css=CSS,
) as demo:

    # --- Header ---
    gr.HTML("""
        <div class="title-block">
          <h1>📄 Research Paper Summarizer</h1>
          <p style="color:#6366f1;font-size:1.05rem;">
            Mistral-7B-Instruct · QLoRA · Fine-tuned on ArXiv + SciTLDR
          </p>
        </div>
        """)

    # --- Main tab ---
    with gr.Tab("✨ Summarize"):
        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(
                    label="Paper Abstract / Introduction",
                    placeholder="Paste your research paper text here …",
                    lines=14,
                    max_lines=30,
                )
                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        64,
                        512,
                        value=256,
                        step=32,
                        label="Max Summary Tokens",
                    )
                    temperature_slider = gr.Slider(
                        0.0,
                        1.0,
                        value=0.1,
                        step=0.05,
                        label="Temperature",
                    )
                summarize_btn = gr.Button("🔍 Summarize", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_summary = gr.Textbox(
                    label="Generated Summary",
                    lines=10,
                    show_copy_button=True,
                )
                status_text = gr.Markdown()

        summarize_btn.click(
            fn=generate_summary,
            inputs=[input_text, max_tokens_slider, temperature_slider],
            outputs=[output_summary, status_text],
        )

        # Examples
        gr.Examples(
            examples=[[ex[1]] for ex in EXAMPLE_ABSTRACTS],
            inputs=[input_text],
            label="📚 Example Papers",
            examples_per_page=2,
        )

    # --- Evaluation tab ---
    with gr.Tab("📊 Evaluate vs. Reference"):
        gr.Markdown("Paste a generated summary and your reference summary to compute ROUGE-L.")
        with gr.Row():
            eval_summary = gr.Textbox(label="Generated Summary", lines=6)
            eval_reference = gr.Textbox(label="Reference Summary", lines=6)
        eval_btn = gr.Button("Compute ROUGE-L", variant="secondary")
        eval_result = gr.Markdown()

        eval_btn.click(
            fn=evaluate_against_reference,
            inputs=[eval_summary, eval_reference],
            outputs=[eval_result],
        )

    # --- About tab ---
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
## About This Model

| | |
|---|---|
| **Base Model** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Fine-tuning Method** | QLoRA (4-bit NF4 quantization) |
| **Training Framework** | [Unsloth](https://github.com/unslothai/unsloth) + TRL SFTTrainer |
| **Training Datasets** | ArXiv Summarization + SciTLDR (AIC split) |
| **LoRA Rank** | r=16, α=32 |

## Evaluation Metrics
- **ROUGE-L** — Longest Common Subsequence recall/precision
- **BERTScore** — Semantic similarity via DeBERTa-xlarge embeddings
- **AlignScore** — Factual consistency (NLI-based alignment)

## Citation
If you use this project, please cite:
```
@misc{research-paper-summarizer-2024,
  title  = {Research Paper Summarizer: Mistral-7B QLoRA},
  author = {Your Name},
  year   = {2024},
  url    = {https://github.com/your-username/research-paper-summarizer}
}
```
            """)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
