# ============================================================
# IMPORTS - Must be at the very top of the file
# ============================================================
import matplotlib.pyplot as plt        # NOT "import matplotlib as plt"
import pandas as pd
import numpy as np
import requests
import json
import os
from datasets import load_dataset, Dataset, DatasetDict

# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
print(f"Dataset loaded: {dataset}")

# ============================================================
# STEP 2: Detect Column Names
# ============================================================
train_df = pd.DataFrame(dataset['train'])
print(f"\nAvailable columns: {list(train_df.columns)}")

src_col = next((c for c in ['article','source','text','document'] if c in train_df.columns), None)
tgt_col = next((c for c in ['highlights','target','summary','abstract'] if c in train_df.columns), None)
print(f"Source column: '{src_col}'")
print(f"Target column: '{tgt_col}'")

# ============================================================
# STEP 3: Calculate Statistics
# ============================================================
source_lengths = [len(item[src_col].split()) for item in dataset['train']]
target_lengths = [len(item[tgt_col].split()) for item in dataset['train']]
compression_ratios = [s/t if t > 0 else 0 for s, t in zip(source_lengths, target_lengths)]

df_stats = pd.DataFrame({
    'Metric': [
        'Mean Source Length (words)',
        'Mean Target Length (words)',
        'Max Source Length (words)',
        'Max Target Length (words)',
        'Min Source Length (words)',
        'Min Target Length (words)',
        'Avg Compression Ratio'
    ],
    'Value': [
        np.mean(source_lengths),
        np.mean(target_lengths),
        np.max(source_lengths),
        np.max(target_lengths),
        np.min(source_lengths),
        np.min(target_lengths),
        np.mean(compression_ratios)
    ]
})
df_stats['Value'] = df_stats['Value'].round(2)
print("\n=== Dataset Statistics ===")
print(df_stats.to_string(index=False))

# ============================================================
# STEP 4: Plot Distributions
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(source_lengths, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Source (Article) Lengths')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].axvline(np.mean(source_lengths), color='red',
                linestyle='--', label=f'Mean: {np.mean(source_lengths):.0f} words')
axes[0].legend()

axes[1].hist(target_lengths, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('Distribution of Target (Summary) Lengths')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].axvline(np.mean(target_lengths), color='red',
                linestyle='--', label=f'Mean: {np.mean(target_lengths):.0f} words')
axes[1].legend()

plt.suptitle('CNN/DailyMail - Length Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
os.makedirs('../docs', exist_ok=True)
plt.savefig('../docs/length_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot saved to docs/length_distributions.png")

# ============================================================
# STEP 5: Data Quality Report
# ============================================================
print("\n=== DATA QUALITY REPORT ===")
print(f"Total training samples : {len(train_df):,}")
print(f"\n--- Missing Values ---")
print(train_df[[src_col, tgt_col]].isnull().sum())

empty_source = (train_df[src_col].str.len() == 0).sum()
empty_target = (train_df[tgt_col].str.len() == 0).sum()
print(f"\n--- Empty Strings ---")
print(f"Empty {src_col} : {empty_source}")
print(f"Empty {tgt_col} : {empty_target}")

if empty_source == 0 and empty_target == 0:
    print("\n✓ Data quality looks good!")
else:
    print(f"\n⚠ Issues found: {empty_source} empty sources, {empty_target} empty targets")