"""
Shared multi-label classification metrics.

Used by both ``scripts/train_text_model.py`` (BERT fine-tuning) and
``scripts/train_image_model.py`` (CLIP linear probe), so text and image
retraining report metrics in an identical shape — macro/micro aggregate
plus a full breakdown per category, as required by the evaluation spec.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, float | int]:
    """
    Compute macro/micro aggregate metrics plus per-label precision, recall,
    f1, and support for a multi-label classification result.

    Args:
        y_true: binary indicator matrix, shape (n_samples, n_labels).
        y_pred: binary indicator matrix of the same shape.
        labels: label names, in the same column order as y_true/y_pred.

    Returns:
        A flat dict with keys: precision_macro, recall_macro, f1_macro,
        precision_micro, recall_micro, f1_micro, and per-label
        precision_<label>, recall_<label>, f1_<label>, support_<label>.
    """
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    metrics: dict[str, float | int] = {
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
    }

    per_label_p, per_label_r, per_label_f1, per_label_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    for i, label in enumerate(labels):
        metrics[f"precision_{label}"] = float(per_label_p[i])
        metrics[f"recall_{label}"] = float(per_label_r[i])
        metrics[f"f1_{label}"] = float(per_label_f1[i])
        metrics[f"support_{label}"] = int(per_label_support[i])

    return metrics


def confusion_matrix_per_label(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """
    Per-label 2x2 confusion matrix (treating each label as an independent
    binary classification problem, appropriate for multi-label output).

    Returns: {label: {"tp": int, "fp": int, "tn": int, "fn": int}, ...}
    """
    result: dict[str, dict[str, int]] = {}
    for i, label in enumerate(labels):
        yt, yp = y_true[:, i], y_pred[:, i]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        result[label] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return result
