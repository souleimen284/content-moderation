"""Tests for app.ml.metrics — shared multi-label metrics used by both trainers."""

from __future__ import annotations

import numpy as np

from app.ml.metrics import confusion_matrix_per_label, multilabel_metrics

LABELS = ["toxic", "spam"]


def test_multilabel_metrics_perfect_predictions():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_pred = y_true.copy()
    metrics = multilabel_metrics(y_true, y_pred, LABELS)
    assert metrics["f1_macro"] == 1.0
    assert metrics["precision_toxic"] == 1.0
    assert metrics["recall_spam"] == 1.0
    assert metrics["support_toxic"] == 2
    assert metrics["support_spam"] == 2


def test_multilabel_metrics_all_wrong():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0, 1], [1, 0]])
    metrics = multilabel_metrics(y_true, y_pred, LABELS)
    assert metrics["f1_macro"] == 0.0
    assert metrics["precision_toxic"] == 0.0


def test_multilabel_metrics_handles_zero_division():
    # No positive predictions at all for "spam" -> precision should be 0,
    # not raise a ZeroDivisionError (zero_division=0 in the underlying call).
    y_true = np.array([[1, 0], [1, 0]])
    y_pred = np.array([[1, 0], [1, 0]])
    metrics = multilabel_metrics(y_true, y_pred, LABELS)
    assert metrics["precision_spam"] == 0.0
    assert metrics["recall_spam"] == 0.0
    assert metrics["support_spam"] == 0


def test_confusion_matrix_per_label_counts():
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    y_pred = np.array([[1, 0], [0, 0], [0, 1], [1, 0]])
    cm = confusion_matrix_per_label(y_true, y_pred, LABELS)

    # toxic: true=[1,1,0,0], pred=[1,0,0,1] -> tp=1, fn=1, tn=1, fp=1
    assert cm["toxic"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    # spam: true=[0,0,1,0], pred=[0,0,1,0] -> tp=1, tn=3, fp=0, fn=0
    assert cm["spam"] == {"tp": 1, "fp": 0, "tn": 3, "fn": 0}


def test_confusion_matrix_counts_sum_to_total_samples():
    y_true = np.random.default_rng(0).integers(0, 2, size=(10, 2))
    y_pred = np.random.default_rng(1).integers(0, 2, size=(10, 2))
    cm = confusion_matrix_per_label(y_true, y_pred, LABELS)
    for label in LABELS:
        total = sum(cm[label].values())
        assert total == 10
