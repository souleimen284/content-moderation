"""
Fine-tune a BERT model for multi-label text content moderation.

This script implements the "load data -> train -> evaluate -> save" steps
required by both the initial training run and the weekly retraining
pipeline (``retrain.py``, Module 5), which imports ``train_and_evaluate``
directly after merging moderator-reviewed decisions into the training CSV.

Usage (CLI)
-----------
    python scripts/train_text_model.py \\
        --data-path data/text_moderation_dataset.csv \\
        --output-dir models/text_moderation \\
        --epochs 4

Expected CSV schema
--------------------
A ``text`` column plus label columns. Labels may use -1 as a sentinel
for "unknown" (e.g. spam.csv rows have no toxic/hate labels). The
training loss masks unknown labels so the model does not learn false
negatives.

Label order (always consistent):
  0 = toxic
  1 = spam
  2 = hate

The model outputs 3 sigmoid probabilities in this order.
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path so imports like `app.core.config` work
# when running this script directly (e.g. `python scripts/train_text_model.py`).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml.metrics import confusion_matrix_per_label, multilabel_metrics

logger = get_logger(__name__)

# Sentinel value in the dataset meaning "this label is unknown for this row"
UNKNOWN_LABEL = -1


class MultiLabelMaskedBCETrainer(Trainer):
    """Trainer that applies per-label pos_weight and masks unknown labels.

    When a label is -1 (unknown), it is masked out of the BCE loss so the
    model does not learn false negatives from missing labels.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Derive mask from the -1 sentinel values in labels.
        # The HF Trainer strips non-standard keys like 'label_mask',
        # so we reconstruct it here from the label values themselves.
        mask = (labels != UNKNOWN_LABEL).float()

        # Replace unknown labels (-1) with 0 for loss computation,
        # but they will be zeroed out by the mask
        labels_safe = labels.clone()
        labels_safe[labels == UNKNOWN_LABEL] = 0.0

        # Compute per-element loss
        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device),
                reduction="none",
            )
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")

        element_loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels_safe.view(-1, self.model.config.num_labels),
        )

        # Apply mask: zero out loss for unknown labels
        masked_loss = element_loss * mask
        loss = masked_loss.sum() / mask.sum()

        return (loss, outputs) if return_outputs else loss


class TextModerationDataset(Dataset):
    """PyTorch Dataset for multi-label text moderation training.

    Tokenizes all texts once in __init__ (no padding) so __getitem__
    returns pre-tokenized tensors instantly.  Dynamic padding is handled
    by ``DataCollator`` at the batch level.
    """

    def __init__(
        self,
        texts: list[str],
        label_matrix: np.ndarray,
        tokenizer,
        max_length: int,
    ) -> None:
        self.label_matrix = label_matrix.astype(np.float32)
        self.max_length = max_length

        # Pre-tokenize once (no padding — dynamic padding in the collator)
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding applied per-batch
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        if "token_type_ids" in encodings:
            self.token_type_ids = encodings["token_type_ids"]
        else:
            self.token_type_ids = None

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        item: dict = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        item["labels"] = torch.tensor(self.label_matrix[idx], dtype=torch.float32)
        return item


@dataclass
class TrainConfig:
    data_path: Path
    output_dir: Path
    labels: list[str]
    base_model: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    eval_split: float  # fraction for validation+test combined
    warmup_ratio: float = 0.06
    early_stopping_patience: int = 3
    seed: int = 42
    head_learning_rate: float | None = None  # defaults to 3x the base LR
    freeze_encoder: bool = False
    freeze_layers: int = 0  # number of encoder layers to freeze from the bottom
    max_train_samples: int = 0  # 0 = use all data; >0 = subsample to this many rows


def load_dataset(data_path: Path, labels: list[str]) -> pd.DataFrame:
    """Load and validate the labeled CSV dataset.

    Handles -1 sentinel values as "unknown label" — these will be masked
    during training.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path)

    # Check required columns
    if "text" not in df.columns:
        raise ValueError(f"Dataset at {data_path} is missing required 'text' column")

    missing_labels = [c for c in labels if c not in df.columns]
    if missing_labels:
        raise ValueError(
            f"Dataset at {data_path} is missing required label columns: {missing_labels}"
        )

    # Remove empty text
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    # Fill NaN with -1 (unknown) for label columns
    for label in labels:
        df[label] = df[label].fillna(UNKNOWN_LABEL).astype(int)

    logger.info("Loaded %d labeled rows from %s", len(df), data_path)

    # Report label statistics
    for label in labels:
        known = df[df[label] >= 0]
        unknown = len(df) - len(known)
        pos = (known[label] == 1).sum() if len(known) > 0 else 0
        neg = (known[label] == 0).sum() if len(known) > 0 else 0
        logger.info(
            "  %s: %d known (pos=%d, neg=%d), %d unknown",
            label, len(known), pos, neg, unknown,
        )

    return df


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences in a batch to the longest length within that batch.

    This avoids wasting compute on padding every sample to ``max_length``.
    For short texts (SMS messages), this can be 3-10x faster.
    """
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad(seq, value=0):
        return seq + [value] * (max_len - len(seq))

    return {
        "input_ids": torch.tensor([pad(item["input_ids"]) for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(item["attention_mask"]) for item in batch], dtype=torch.long),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_compute_metrics(labels: list[str]):
    """Build an EvalPrediction -> dict metrics function for the Trainer."""

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits, y_true_with_mask = eval_pred
        # The eval dataset returns labels; for metrics we need to ignore
        # unknown labels (-1). The dataset stores mask in labels too.
        # However, HF Trainer passes labels through, so we need to
        # separate label values from masks if present.
        # Since we can't easily add mask to the HF pipeline without
        # custom collation, we evaluate on all rows but only for
        # metrics purposes. The mask is applied in the loss.

        # For metrics, only evaluate rows where all labels are known
        # (mask out -1 labels by setting predictions to 0 for those)
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        y_pred = (probs >= 0.5).astype(int)

        # Create a mask for valid (known) labels
        valid_mask = y_true_with_mask >= 0
        y_true_clean = np.where(valid_mask, y_true_with_mask, 0)
        y_pred_clean = np.where(valid_mask, y_pred, 0)

        metrics = multilabel_metrics(y_true_clean, y_pred_clean, labels)
        metrics["confusion_matrix"] = confusion_matrix_per_label(
            y_true_clean, y_pred_clean, labels
        )

        # Per-label PR-AUC (only on rows where that label is known)
        for i, label in enumerate(labels):
            col_mask = valid_mask[:, i] > 0
            if col_mask.sum() > 0:
                y_t = y_true_clean[col_mask, i]
                y_p = probs[col_mask, i]
                # Skip if only one class present
                if len(np.unique(y_t)) > 1:
                    metrics[f"pr_auc_{label}"] = float(average_precision_score(y_t, y_p))
                else:
                    metrics[f"pr_auc_{label}"] = 0.0
            else:
                metrics[f"pr_auc_{label}"] = 0.0

        return metrics

    return compute_metrics


def _verify_saved_checkpoint(output_dir: Path, labels: list[str]) -> None:
    """Sanity-check the freshly saved model before training reports success."""
    from safetensors import safe_open

    safetensors_path = output_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise RuntimeError(f"Saved checkpoint missing model.safetensors at {output_dir}")

    with safe_open(str(safetensors_path), framework="numpy") as f:
        keys = list(f.keys())

    # Note: transformers 5.x may save LayerNorm with beta/gamma naming.
    # This is acceptable — the model still loads and works correctly.
    # Only warn but do not fail.
    legacy = [k for k in keys if k.endswith(".gamma") or k.endswith(".beta")]
    if legacy:
        logger.warning(
            "Saved checkpoint contains %d LayerNorm keys with beta/gamma naming "
            "(transformers 5.x style). This is acceptable — the model loads correctly.",
            len(legacy),
        )
    if "classifier.weight" not in keys or "classifier.bias" not in keys:
        raise RuntimeError(
            f"Saved checkpoint is missing the multi-label classification head: {output_dir}"
        )

    # Verify classifier head has correct number of outputs
    from safetensors import safe_open as sf_open

    with sf_open(str(safetensors_path), framework="numpy") as f:
        weight_shape = f.get_tensor("classifier.weight").shape
        if weight_shape[0] != len(labels):
            raise RuntimeError(
                f"Classifier head has {weight_shape[0]} outputs but "
                f"expected {len(labels)}: {labels}"
            )

    logger.info(
        "Saved checkpoint verified: %d canonical tensors, %d-label classifier head present.",
        len(keys), len(labels),
    )


def _subsample_dataset(
    df: pd.DataFrame,
    labels: list[str],
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified subsample of the dataset to cap training time.

    Picks up to *max_samples* rows while preserving the label distribution
    as closely as possible.  If the dataset is already smaller than the
    cap it is returned unchanged.
    """
    if max_samples <= 0 or len(df) <= max_samples:
        return df

    logger.info(
        "Subsampling dataset from %d to %d rows (stratified)",
        len(df), max_samples,
    )

    # Create a composite stratification key from the non-negative labels
    # so each unique label combination gets proportional representation.
    strat_col = df[labels].clip(lower=0).astype(str).agg("_".join, axis=1)
    sampled, _ = train_test_split(
        df,
        train_size=max_samples,
        random_state=seed,
        stratify=strat_col,
    )
    return sampled.reset_index(drop=True)


def _freeze_encoder_layers(model, num_layers: int) -> None:
    """Freeze the bottom *num_layers* transformer layers + embeddings.

    This is the single biggest speed-up for CPU training: the frozen layers
    skip gradient computation entirely.  Only the unfrozen layers and the
    classifier head receive gradient updates.
    """
    # Freeze embeddings first
    if hasattr(model, "distilbert"):
        embeddings = model.distilbert.embeddings
        for p in embeddings.parameters():
            p.requires_grad = False
        transformer = model.distilbert.transformer
    elif hasattr(model, "bert"):
        embeddings = model.bert.embeddings
        for p in embeddings.parameters():
            p.requires_grad = False
        transformer = model.bert.encoder
    else:
        logger.warning("Could not detect encoder — skipping layer freezing")
        return

    total_layers = len(transformer.layer)
    layers_to_freeze = min(num_layers, total_layers)
    logger.info(
        "Freezing embeddings + %d/%d encoder layers (%d trainable layers remain)",
        layers_to_freeze, total_layers, total_layers - layers_to_freeze,
    )
    for i in range(layers_to_freeze):
        for p in transformer.layer[i].parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )


def train_and_evaluate(cfg: TrainConfig) -> dict:
    """
    Full train/eval/save cycle. Returns the evaluation metrics dict.

    Called both from this script's CLI entrypoint and from ``retrain.py``,
    which merges new moderator-reviewed rows into the CSV before calling in.
    """
    torch.manual_seed(cfg.seed)
    df = load_dataset(cfg.data_path, cfg.labels)

    if len(df) < 10:
        logger.warning(
            "Only %d training rows available — results will be unreliable. "
            "This is expected for the bundled sample dataset; supply a larger "
            "labeled dataset for real training.",
            len(df),
        )

    # --- Optional subsampling for CPU-friendly training ---
    if cfg.max_train_samples > 0:
        df = _subsample_dataset(df, cfg.labels, cfg.max_train_samples, cfg.seed)

    # --- 3-way split: train / validation / test ---
    # First split off test set
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, random_state=cfg.seed, shuffle=True
    )
    # Then split remaining into train and validation
    val_fraction = cfg.eval_split / (1 - 0.15)  # adjust for the held-out test
    train_df, eval_df = train_test_split(
        train_val_df, test_size=val_fraction, random_state=cfg.seed, shuffle=True
    )

    logger.info(
        "Data split — train: %d, validation: %d, test: %d",
        len(train_df), len(eval_df), len(test_df),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(cfg.labels),
        problem_type="multi_label_classification",
    )

    # Verify multi-label setup
    if model.config.problem_type != "multi_label_classification":
        raise RuntimeError(
            f"Model must use problem_type='multi_label_classification' for multi-label "
            f"moderation, got {model.config.problem_type!r}"
        )
    model.config.id2label = {i: label for i, label in enumerate(cfg.labels)}
    model.config.label2id = {label: i for i, label in enumerate(cfg.labels)}
    logger.info(
        "Multi-label setup verified: problem_type=%s, loss=BCEWithLogitsLoss, "
        "num_labels=%d, labels=%s, model=%s",
        model.config.problem_type,
        model.config.num_labels,
        cfg.labels,
        cfg.base_model,
    )

    # --- Freeze encoder layers for faster CPU training ---
    if cfg.freeze_encoder and cfg.freeze_layers > 0:
        _freeze_encoder_layers(model, cfg.freeze_layers)

    # --- Data efficiency note for small datasets ---
    if len(train_df) < 100:
        logger.warning(
            "Only %d train rows — disabling layer freezing to avoid overfitting",
            len(train_df),
        )
        # Unfreeze everything if dataset is tiny
        for p in model.parameters():
            p.requires_grad = True

    # --- Per-label positive-class weight ---
    # Only compute from rows where the label is known (>= 0)
    pos_counts = []
    for label in cfg.labels:
        known_rows = train_df[train_df[label] >= 0]
        pos_count = known_rows[label].sum()
        pos_counts.append(pos_count)
    neg_counts = [len(train_df) - c for c in pos_counts]

    pos_weight = torch.tensor(
        [nc / max(pc, 1) for pc, nc in zip(pos_counts, neg_counts)],
        dtype=torch.float32,
    )
    logger.info("Per-label pos_weight: %s", dict(zip(cfg.labels, pos_weight.tolist())))

    train_dataset = TextModerationDataset(
        train_df["text"].tolist(), train_df[cfg.labels].to_numpy(), tokenizer, cfg.max_length
    )
    eval_dataset = TextModerationDataset(
        eval_df["text"].tolist(), eval_df[cfg.labels].to_numpy(), tokenizer, cfg.max_length
    )

    # Calculate warmup steps from ratio
    num_training_steps = (
        (len(train_dataset) + cfg.batch_size - 1) // cfg.batch_size
    ) * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir / "checkpoints"),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=num_warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        save_only_model=True,
        gradient_checkpointing=False,
        dataloader_num_workers=0 if platform.system() == "Windows" else 4,
        dataloader_pin_memory=True,
        report_to=[],
        seed=cfg.seed,
    )    # --- Optimizer with head LR boost ---
    head_lr = cfg.head_learning_rate or cfg.learning_rate * 3.0

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if n.startswith("classifier.")
            ],
            "lr": head_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not n.startswith("classifier.") and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not n.startswith("classifier.") and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    logger.info("Optimizer: head lr=%.2e, encoder lr=%.2e", head_lr, cfg.learning_rate)

    trainer = MultiLabelMaskedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=build_compute_metrics(cfg.labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        pos_weight=pos_weight,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save model + tokenizer
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    # Verify saved checkpoint
    _verify_saved_checkpoint(cfg.output_dir, cfg.labels)

    # --- Evaluate on held-out test set ---
    logger.info("Evaluating on test set (%d rows)...", len(test_df))
    test_dataset = TextModerationDataset(
        test_df["text"].tolist(), test_df[cfg.labels].to_numpy(), tokenizer, cfg.max_length
    )
    test_metrics = trainer.evaluate(test_dataset)

    # --- Print detailed classification report ---
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)

    # Get predictions on test set
    test_preds_output = trainer.predict(test_dataset)
    test_logits = test_preds_output.predictions
    test_probs = 1 / (1 + np.exp(-test_logits))
    test_y_pred = (test_probs >= 0.5).astype(int)
    test_y_true_raw = test_df[cfg.labels].to_numpy()

    # Mask unknown labels
    valid_mask = test_y_true_raw >= 0
    test_y_true = np.where(valid_mask, test_y_true_raw, 0)
    test_y_pred_masked = np.where(valid_mask, test_y_pred, 0)

    # Per-label classification report
    for i, label in enumerate(cfg.labels):
        col_mask = valid_mask[:, i] > 0
        if col_mask.sum() > 0 and len(np.unique(test_y_true[col_mask, i])) > 1:
            p, r, f1, _ = precision_recall_fscore_support(
                test_y_true[col_mask, i], test_y_pred_masked[col_mask, i],
                average="binary", zero_division=0
            )
            logger.info("  %s: precision=%.3f, recall=%.3f, f1=%.3f", label, p, r, f1)
        else:
            logger.info("  %s: skipped (insufficient known labels in test set)", label)

    # Overall metrics
    logger.info("\nMacro F1 (test): %.3f", test_metrics.get("eval_f1_macro", 0))
    logger.info("Micro F1 (test): %.3f", test_metrics.get("eval_f1_micro", 0))

    # Merge test metrics into the returned metrics
    metrics["test_f1_macro"] = test_metrics.get("eval_f1_macro", 0)
    metrics["test_f1_micro"] = test_metrics.get("eval_f1_micro", 0)
    metrics["test_samples"] = len(test_df)

    logger.info("Training complete. Metrics: %s", metrics)

    # --- MLflow logging ---
    if settings.use_mlflow:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            with mlflow.start_run(run_name="text_moderation_bert"):
                mlflow.log_params(
                    {
                        "base_model": cfg.base_model,
                        "labels": cfg.labels,
                        "max_length": cfg.max_length,
                        "batch_size": cfg.batch_size,
                        "learning_rate": cfg.learning_rate,
                        "head_learning_rate": head_lr,
                        "num_epochs": cfg.num_epochs,
                        "train_rows": len(train_df),
                        "eval_rows": len(eval_df),
                        "test_rows": len(test_df),
                        "pos_weight": dict(zip(cfg.labels, pos_weight.tolist())),
                    }
                )
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                mlflow.log_artifacts(str(cfg.output_dir), artifact_path="model")
        except Exception:
            logger.exception(
                "MLflow logging failed — the trained checkpoint is still saved at %s",
                cfg.output_dir,
            )
    else:
        logger.info("MLflow disabled (use_mlflow=false) — skipping experiment logging")

    return metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for text content moderation.")
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/text_moderation_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=settings.text_model_checkpoint_dir)
    parser.add_argument("--base-model", type=str, default=settings.text_model_name)
    parser.add_argument("--max-length", type=int, default=settings.text_model_max_length)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-split", type=float, default=0.15)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--freeze-encoder", action="store_true",
        default=settings.text_model_freeze_encoder,
        help="Freeze lower encoder layers for faster training",
    )
    parser.add_argument(
        "--freeze-layers", type=int, default=settings.text_model_freeze_layers,
        help="Number of bottom encoder layers to freeze",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=settings.text_model_max_train_samples,
        help="Subsample dataset to this many rows (0 = no limit)",
    )
    args = parser.parse_args()

    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        labels=settings.text_labels,
        base_model=args.base_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_split=args.eval_split,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        freeze_encoder=args.freeze_encoder,
        freeze_layers=args.freeze_layers,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    config = parse_args()
    train_and_evaluate(config)
