import torch
from transformers import pipeline

# Check if MPS (Mac GPU) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load a simple summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

text = """
The Apollo program, also known as Project Apollo, was the third United States human spaceflight program 
carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing 
and landing the first humans on the Moon from 1968 to 1972.
"""

summary = summarizer(text, max_length=30, min_length=10, do_sample=False)
print(f"Summary: {summary[0]['summary_text']}")
