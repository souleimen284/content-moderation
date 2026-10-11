"""
Train a multi-label linear probe on frozen CLIP image embeddings.

CLIP is a contrastively pretrained image-text model, not a classifier —
fine-tuning its own weights well requires large paired image-text data
that moderator review alone doesn't provide. Instead, this script adapts
the image moderator to labeled examples the standard way for limited data:
freeze CLIP's image encoder, extract embeddings, and train a lightweight
multi-label classifier (``OneVsRestClassifier(LogisticRegression)``) on
top. This is what ``ImageModerationModel`` uses at inference time when a
trained probe is present (see ``app.ml.image_moderation``).

Called by ``retrain.py`` after exporting moderator-reviewed image requests
from the database (``queue_service.export_moderator_reviewed_images_for_retraining``).

Usage (CLI)
-----------
    python scripts/train_image_model.py --output-dir models/image_moderation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from app.core.config import settings
from app.core.logging_config import get_logger
from app.db.database import session_scope
from app.ml.image_moderation import ImageModerationModel
from app.ml.metrics import confusion_matrix_per_label, multilabel_metrics
from app.services.queue_service import export_moderator_reviewed_images_for_retraining

logger = get_logger(__name__)


@dataclass
class ImageTrainConfig:
    output_dir: Path
    labels: list[str] = field(default_factory=lambda: list(settings.image_labels))
    eval_split: float = 0.2
    min_samples: int = 10
    seed: int = 42


def load_review_data(labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Pull moderator-reviewed image paths + label matrix from the database."""
    with session_scope() as db:
        rows = export_moderator_reviewed_images_for_retraining(db, labels)

    if not rows:
        return [], np.empty((0, len(labels)))

    image_paths = [r["image_path"] for r in rows]
    label_matrix = np.array([r["labels"] for r in rows], dtype=int)
    return image_paths, label_matrix


def train_and_evaluate(cfg: ImageTrainConfig) -> dict | None:
    """
    Full train/eval/save cycle for the CLIP linear probe.

    Returns the evaluation metrics dict, or ``None`` if there wasn't enough
    moderator-reviewed image data to train on (a legitimate, non-error
    outcome — retrain.py treats this as "skip this modality this week").
    """
    image_paths, label_matrix = load_review_data(cfg.labels)

    if len(image_paths) < cfg.min_samples:
        logger.warning(
            "Only %d moderator-reviewed images available (need >= %d) — "
            "skipping image linear-probe retraining this cycle.",
            len(image_paths),
            cfg.min_samples,
        )
        return None

    logger.info("Training image linear probe on %d moderator-reviewed images", len(image_paths))

    # Use the base (zero-shot) CLIP encoder to extract frozen embeddings —
    # we're training a classifier on top, not touching CLIP's own weights.
    encoder = ImageModerationModel.from_pretrained(labels=cfg.labels)
    embeddings = encoder.encode_images(image_paths)

    X_train, X_eval, y_train, y_eval = train_test_split(
        embeddings, label_matrix, test_size=cfg.eval_split, random_state=cfg.seed
    )

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.seed)
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_eval)
    metrics = multilabel_metrics(y_eval, y_pred, cfg.labels)
    metrics["confusion_matrix"] = confusion_matrix_per_label(y_eval, y_pred, cfg.labels)

    if settings.use_mlflow:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            with mlflow.start_run(run_name="image_moderation_clip_probe"):
                mlflow.log_params(
                    {
                        "base_model": settings.image_model_name,
                        "pretrained": settings.image_model_pretrained,
                        "labels": cfg.labels,
                        "train_samples": len(X_train),
                        "eval_samples": len(X_eval),
                        "classifier": "OneVsRestClassifier(LogisticRegression)",
                    }
                )
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

                probe_path = ImageModerationModel.save_linear_probe(classifier, cfg.output_dir)
                mlflow.log_artifact(str(probe_path), artifact_path="model")
        except Exception:
            logger.exception("MLflow logging failed")
    else:
        logger.info("MLflow disabled (use_mlflow=false) — skipping experiment logging")

    logger.info("Image linear-probe training complete. Metrics: %s", metrics)
    metrics["_training_samples"] = len(image_paths)
    return metrics


def parse_args() -> ImageTrainConfig:
    parser = argparse.ArgumentParser(description="Train a CLIP linear probe for image moderation.")
    parser.add_argument("--output-dir", type=Path, default=settings.image_model_checkpoint_dir)
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--min-samples", type=int, default=10)
    args = parser.parse_args()

    return ImageTrainConfig(
        output_dir=args.output_dir,
        eval_split=args.eval_split,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    config = parse_args()
    train_and_evaluate(config)
