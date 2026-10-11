#!/usr/bin/env python3
"""
Build the unified text moderation dataset from two source datasets.

Source datasets:
  A) data/toxic_hate.csv  — Jigsaw-style toxicity dataset with columns:
       id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
  B) data/spam.csv  — SMS Spam Collection with columns:
       v1 (spam/ham), v2 (text)

Output: data/text_moderation_dataset.csv with columns:
       text, toxic, spam, hate

Label construction:
  toxic_hate.csv →
    TOXIC: max(toxic, severe_toxic, insult) — any of these Jigsaw labels
           indicates toxicity. We take the maximum to capture a broader
           notion of toxicity (a comment can be severely toxic OR insulting,
           both count as "toxic" for our purposes).
    HATE:  max(identity_hate, threat) — identity_hate directly targets
           identity groups (hate speech); threats against groups are also
           a form of hate.
    SPAM:  Unknown for this dataset → UNKNOWN (masked out during training)

  spam.csv →
    SPAM: 1 if v1=="spam", 0 if v1=="ham"
    TOXIC: Unknown → UNKNOWN
    HATE:  Unknown → UNKNOWN

Mask strategy:
  Unknown labels use -1 as a sentinel. During training, the loss function
  masks out unknown labels so the model does not learn false negatives.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_from_toxic_hate(path: Path) -> pd.DataFrame:
    """Convert toxic_hate.csv into our unified schema with known labels."""
    df = pd.read_csv(path)

    # Standardize column names
    text_col = "comment_text" if "comment_text" in df.columns else "text"
    df = df.rename(columns={text_col: "text"})

    # --- TOXIC label ---
    # Use max(toxic, severe_toxic, insult) as composite toxicity signal.
    toxic_cols = [c for c in ["toxic", "severe_toxic", "insult"] if c in df.columns]
    df["toxic"] = df[toxic_cols].max(axis=1).fillna(0).astype(int) if toxic_cols else 0

    # --- HATE label ---
    # Use max(identity_hate, threat) as hate speech signal.
    hate_cols = [c for c in ["identity_hate", "threat"] if c in df.columns]
    df["hate"] = df[hate_cols].max(axis=1).fillna(0).astype(int) if hate_cols else 0

    # --- SPAM: Unknown → sentinel -1 ---
    df["spam"] = -1

    return df[["text", "toxic", "hate", "spam"]].copy()


def build_from_spam(path: Path) -> pd.DataFrame:
    """Convert spam.csv into our unified schema with known labels."""
    # spam.csv is encoded in latin-1 and may have extra unnamed columns
    df = pd.read_csv(path, encoding="latin-1")

    text_col = "v2" if "v2" in df.columns else "text"
    label_col = "v1" if "v1" in df.columns else "label"

    df = df.rename(columns={text_col: "text", label_col: "label_type"})
    df = df[["text", "label_type"]].copy()

    # SPAM label: ham → 0, spam → 1
    df["spam"] = (df["label_type"].str.lower() == "spam").astype(int)

    # TOXIC and HATE: Unknown → sentinel -1
    df["toxic"] = -1
    df["hate"] = -1

    return df[["text", "toxic", "hate", "spam"]].copy()


def report_stats(df: pd.DataFrame, name: str) -> None:
    """Print dataset statistics."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(df)}")

    for label in ["toxic", "hate", "spam"]:
        if label not in df.columns:
            continue
        known = df[label] >= 0
        unknown_count = (~known).sum()
        known_df = df[known]
        if len(known_df) > 0:
            pos_count = (known_df[label] == 1).sum()
            neg_count = (known_df[label] == 0).sum()
            print(f"  {label}: {len(known_df)} known (pos={pos_count}, neg={neg_count}), {unknown_count} unknown")
        else:
            print(f"  {label}: all {unknown_count} rows unknown")


def build_unified_dataset(
    toxic_hate_path: Path,
    spam_path: Path,
    output_path: Path,
) -> None:
    """Build and save the unified text moderation dataset."""
    toxic_df = build_from_toxic_hate(toxic_hate_path)
    spam_df = build_from_spam(spam_path)

    report_stats(toxic_df, "toxic_hate.csv")
    report_stats(spam_df, "spam.csv")

    # --- Data quality checks ---
    toxic_df = toxic_df[toxic_df["text"].notna() & (toxic_df["text"].str.strip() != "")]
    spam_df = spam_df[spam_df["text"].notna() & (spam_df["text"].str.strip() != "")]

    toxic_df = toxic_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    spam_df = spam_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    # --- Handle cross-source deduplication ---
    toxic_texts = set(toxic_df["text"])
    spam_texts = set(spam_df["text"])
    overlap = toxic_texts & spam_texts

    if overlap:
        print(f"\n{len(overlap)} texts appear in both datasets. Merging labels...")
        overlap_spam = spam_df[spam_df["text"].isin(overlap)].set_index("text")
        for text in overlap:
            row_idx = toxic_df[toxic_df["text"] == text].index
            if len(row_idx) > 0 and text in overlap_spam.index:
                toxic_df.loc[row_idx[0], "spam"] = overlap_spam.loc[text, "spam"]
        spam_df = spam_df[~spam_df["text"].isin(overlap)].reset_index(drop=True)

    # --- Concatenate ---
    unified = pd.concat([toxic_df, spam_df], ignore_index=True)
    unified = unified.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print("Unified Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(unified)}")
    report_stats(unified, "unified")

    unified.to_csv(output_path, index=False)
    print(f"\nSaved unified dataset to {output_path}")
    print(f"Columns: {list(unified.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Build unified text moderation dataset.")
    parser.add_argument("--toxic-hate-path", type=Path, default=Path("data/toxic_hate.csv"))
    parser.add_argument("--spam-path", type=Path, default=Path("data/spam.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/text_moderation_dataset.csv"))
    args = parser.parse_args()

    build_unified_dataset(args.toxic_hate_path, args.spam_path, args.output_path)


if __name__ == "__main__":
    main()
