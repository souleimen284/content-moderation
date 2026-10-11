#!/usr/bin/env python3
"""
Create a clean, balanced final training dataset from toxic_hate.csv and spam.csv.

Features:
- Removes duplicates and empty rows
- Balanced representation of all 3 labels (toxic, spam, hate)
- Reasonable size for CPU training (~16K rows)
- Contains good diversity of label combinations
"""

import pandas as pd
from pathlib import Path

print("Loading source datasets...")
toxic_hate = pd.read_csv('data/toxic_hate.csv')
spam = pd.read_csv('data/spam.csv', encoding='latin-1')

print(f"toxic_hate.csv: {len(toxic_hate)} rows")
print(f"spam.csv: {len(spam)} rows")

# === TOXIC/HATE dataset processing ===
# Standardize column names
if 'comment_text' in toxic_hate.columns:
    toxic_hate = toxic_hate.rename(columns={'comment_text': 'text'})

# Create composite labels
toxic_cols = [c for c in ['toxic', 'severe_toxic', 'insult'] if c in toxic_hate.columns]
hate_cols = [c for c in ['identity_hate', 'threat'] if c in toxic_hate.columns]

toxic_hate['toxic'] = toxic_hate[toxic_cols].max(axis=1).fillna(0).astype(int) if toxic_cols else 0
toxic_hate['hate'] = toxic_hate[hate_cols].max(axis=1).fillna(0).astype(int) if hate_cols else 0
toxic_hate['spam'] = -1  # unknown

toxic_hate = toxic_hate[['text', 'toxic', 'hate', 'spam']].copy()

# === SPAM dataset processing ===
spam_cols = {col: 'spam' if col in ['v1', 'label'] else col for col in spam.columns}
spam = spam.rename(columns=spam_cols)
text_col = 'v2' if 'v2' in spam.columns else ('text' if 'text' in spam.columns else spam.columns[1])
spam = spam.rename(columns={text_col: 'text'})

if 'spam' in spam.columns and spam['spam'].dtype == 'object':
    spam['spam'] = (spam['spam'].str.lower().str.strip() == 'spam').astype(int)
else:
    spam['spam'] = spam['spam'].astype(int) if 'spam' in spam.columns else 1

spam['toxic'] = -1
spam['hate'] = -1
spam = spam[['text', 'toxic', 'hate', 'spam']].copy()

# === DATA CLEANING ===
print("\nCleaning data...")

# Remove empty/null text
toxic_hate = toxic_hate[toxic_hate['text'].notna() & (toxic_hate['text'].str.strip() != '')].reset_index(drop=True)
spam = spam[spam['text'].notna() & (spam['text'].str.strip() != '')].reset_index(drop=True)

# Remove duplicates
toxic_hate = toxic_hate.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
spam = spam.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

# Handle cross-dataset deduplication
overlap = set(toxic_hate['text']) & set(spam['text'])
if overlap:
    print(f"Found {len(overlap)} overlapping texts")
    spam_overlap = spam[spam['text'].isin(overlap)].set_index('text')
    for text in overlap:
        idx = toxic_hate[toxic_hate['text'] == text].index
        if len(idx) > 0 and text in spam_overlap.index:
            toxic_hate.loc[idx[0], 'spam'] = spam_overlap.loc[text, 'spam']
    spam = spam[~spam['text'].isin(overlap)].reset_index(drop=True)

print(f"After cleaning:")
print(f"  toxic_hate: {len(toxic_hate)} rows")
print(f"  spam: {len(spam)} rows")

# === BALANCED SAMPLING ===
print("\nCreating balanced sample...")

# Sample strategy:
# - From toxic_hate: prioritize rows with known labels (not -1)
# - Include some with only toxic, some with only hate, some with both
# - From spam: all rows with spam=1

# Toxic_hate sampling: 12000 rows
known_toxic_hate = toxic_hate[(toxic_hate['toxic'] >= 0) | (toxic_hate['hate'] >= 0)].copy()
unknown_toxic_hate = toxic_hate[(toxic_hate['toxic'] < 0) & (toxic_hate['hate'] < 0)].copy()

# Prioritize known labels
if len(known_toxic_hate) > 12000:
    # Simple random sample from known labels (keeps diversity)
    sampled_toxic_hate = known_toxic_hate.sample(n=12000, random_state=42).reset_index(drop=True)
else:
    sampled_toxic_hate = known_toxic_hate.copy()
    if len(sampled_toxic_hate) < 12000:
        remaining = 12000 - len(sampled_toxic_hate)
        sampled_toxic_hate = pd.concat([
            sampled_toxic_hate,
            unknown_toxic_hate.sample(min(len(unknown_toxic_hate), remaining), random_state=42)
        ]).reset_index(drop=True)

# Spam sampling: all of it (typically 4-5K)
sampled_spam = spam.copy()

print(f"Sampled:")
print(f"  from toxic_hate: {len(sampled_toxic_hate)} rows")
print(f"  from spam: {len(sampled_spam)} rows")

# === MERGE ===
unified = pd.concat([sampled_toxic_hate, sampled_spam], ignore_index=True)
unified = unified.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

print(f"\nFinal unified dataset: {len(unified)} rows")

# === LABEL DISTRIBUTION ===
print("\nLabel distribution:")
for label in ['toxic', 'hate', 'spam']:
    known = unified[unified[label] >= 0]
    unknown = len(unified) - len(known)
    if len(known) > 0:
        pos = (known[label] == 1).sum()
        neg = (known[label] == 0).sum()
        print(f"  {label}: {len(known)} known (pos={pos}, neg={neg}), {unknown} unknown")
    else:
        print(f"  {label}: all {unknown} unknown")

# === SAVE ===
output_path = 'data/text_moderation_final_dataset.csv'
unified.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"Columns: {list(unified.columns)}")
print("\nYou can now train with:")
print("  python -m scripts.train_text_model \\")
print("    --data-path data/text_moderation_final_dataset.csv \\")
print("    --output-dir models/text_moderation \\")
print("    --epochs 4")
