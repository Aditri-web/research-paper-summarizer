import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

print("Loading dataset...")
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=False)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

available_cols = dataset["train"].column_names
print("Available columns:", available_cols)
if "source" in available_cols:
    src_col = "source"
    tgt_col = "target"
elif "article" in available_cols:
    src_col = "article"
    tgt_col = "highlights"
else:
    raise ValueError(f"Could not find correct columns. Available: {available_cols}")

sample_size = 5
subset = dataset["train"].select(range(sample_size))

print("Testing tokenization...")
for item in subset:
    print("Article len:", len(item[src_col]))
    tokens = tokenizer.encode(item[src_col])
    print("Tokens count:", len(tokens))
