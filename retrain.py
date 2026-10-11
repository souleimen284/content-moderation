#!/usr/bin/env python3
"""
Weekly retraining pipeline orchestrator.

Runs the full retrain cycle for both moderation models:

  1. Export moderator decisions from the database (Module 5's queue service).
  2. Merge them into the training data (text: append to the labeled CSV;
     image: pulled directly from the DB by the image trainer).
  3. Retrain each model (BERT fine-tune / CLIP linear probe).
  4. Evaluate the newly trained model against the currently active one.
  5. Promote the new model to active only if it's actually better
     (app.services.model_registry.promote_if_better); otherwise the
     previous model stays in production and the attempt is logged.
  6. Version everything in the ``model_versions`` table and write an
     audit log entry summarizing the run.

Intended to run on a schedule (``settings.retrain_schedule_cron``, default
weekly) via cron / a Kubernetes CronJob / Airflow — see docker/ and the
README for deployment options. Can also be run manually:

    python retrain.py
    python retrain.py --skip-image
    python retrain.py --text-data-path data/sample_text_moderation.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.core.logging_config import get_logger
from app.db.database import session_scope
from app.db.models import AuditLog, Modality
from app.ml.text_moderation import reload_text_moderation_model
from app.ml.image_moderation import reload_image_moderation_model
from app.services import model_registry
from app.services.queue_service import export_moderator_reviewed_text_for_retraining
from scripts.train_text_model import TrainConfig, train_and_evaluate as train_text
from scripts.train_image_model import ImageTrainConfig, train_and_evaluate as train_image

logger = get_logger(__name__)


@dataclass
class RetrainRunResult:
    modality: str
    ran: bool
    promoted: bool = False
    metrics: dict | None = None
    version_tag: str | None = None
    reason: str | None = None


def _version_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def merge_text_training_data(base_data_path: Path) -> tuple[Path, int]:
    """
    Merge moderator-reviewed text decisions from the DB into the base
    labeled CSV, deduplicating on exact text match (a moderator re-review
    of previously-seen text overrides the earlier label).

    Returns the path to the merged CSV (written under
    ``settings.retrain_data_dir``) and the number of new reviewed rows
    that were merged in.
    """
    with session_scope() as db:
        new_rows = export_moderator_reviewed_text_for_retraining(db, settings.text_labels)

    base_df = pd.read_csv(base_data_path) if base_data_path.exists() else pd.DataFrame(
        columns=["text", *settings.text_labels]
    )

    if new_rows.empty:
        merged = base_df
    else:
        # New moderator-reviewed rows take precedence over the base dataset
        # for any text that appears in both (moderator correction wins).
        merged = pd.concat([base_df, new_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset="text", keep="last")

    settings.retrain_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.retrain_data_dir / f"text_retrain_{_version_tag()}.csv"
    merged.to_csv(output_path, index=False)

    logger.info(
        "Merged %d moderator-reviewed rows into base dataset (%d base rows) -> %d total rows at %s",
        len(new_rows),
        len(base_df),
        len(merged),
        output_path,
    )
    return output_path, len(new_rows)


def run_text_retraining(base_data_path: Path, min_new_samples: int) -> RetrainRunResult:
    """Merge data, fine-tune BERT, evaluate, and promote if better."""
    merged_path, new_row_count = merge_text_training_data(base_data_path)

    if new_row_count < min_new_samples:
        logger.warning(
            "Only %d new moderator-reviewed text rows since last retrain (need >= %d). "
            "Proceeding anyway using all available labeled data, but flagging low novelty.",
            new_row_count,
            min_new_samples,
        )

    version_tag = _version_tag()
    output_dir = Path(settings.text_model_checkpoint_dir).parent / f"text_moderation_{version_tag}"

    cfg = TrainConfig(
        data_path=merged_path,
        output_dir=output_dir,
        labels=settings.text_labels,
        base_model=settings.text_model_name,
        max_length=settings.text_model_max_length,
        batch_size=8,  # modest batch keeps fine-tuning RAM well under the Docker VM ceiling
        learning_rate=1.5e-5,
        head_learning_rate=1e-4,  # head trains 6.7x faster than the encoder body
        num_epochs=6,
        weight_decay=0.01,
        eval_split=0.15,
    )
    metrics = train_text(cfg)

    with session_scope() as db:
        active_version, promoted = model_registry.promote_if_better(
            db,
            model_type=Modality.TEXT,
            candidate_version_tag=version_tag,
            candidate_checkpoint_path=str(output_dir),
            candidate_metrics=metrics,
            training_samples=new_row_count,
        )

    if promoted:
        # Point the "live" checkpoint dir the API loads from at the new
        # version, then force the in-process singleton to reload it. A
        # reload failure here does NOT undo the promotion — the DB record
        # and on-disk checkpoint are already updated; at worst this worker
        # process keeps serving the previous model in memory until its next
        # restart or an explicit reload call.
        _promote_checkpoint(output_dir, Path(settings.text_model_checkpoint_dir))
        try:
            reload_text_moderation_model()
        except Exception:
            logger.exception(
                "Text model promoted to %s but in-process hot-reload failed; "
                "this worker will keep serving the previous model until restarted.",
                version_tag,
            )

    return RetrainRunResult(
        modality="text", ran=True, promoted=promoted, metrics=metrics, version_tag=version_tag
    )


def run_image_retraining(min_samples: int) -> RetrainRunResult:
    """Train the CLIP linear probe, evaluate, and promote if better."""
    version_tag = _version_tag()
    output_dir = Path(settings.image_model_checkpoint_dir).parent / f"image_moderation_{version_tag}"

    cfg = ImageTrainConfig(
        output_dir=output_dir,
        labels=settings.image_labels,
        eval_split=0.2,
        min_samples=min_samples,
    )
    metrics = train_image(cfg)

    if metrics is None:
        return RetrainRunResult(
            modality="image", ran=False, reason="insufficient moderator-reviewed image data"
        )

    training_samples = metrics.pop("_training_samples", 0)

    with session_scope() as db:
        active_version, promoted = model_registry.promote_if_better(
            db,
            model_type=Modality.IMAGE,
            candidate_version_tag=version_tag,
            candidate_checkpoint_path=str(output_dir),
            candidate_metrics=metrics,
            training_samples=training_samples,
        )

    if promoted:
        _promote_checkpoint(output_dir, Path(settings.image_model_checkpoint_dir))
        try:
            reload_image_moderation_model()
        except Exception:
            logger.exception(
                "Image model promoted to %s but in-process hot-reload failed; "
                "this worker will keep serving the previous model until restarted.",
                version_tag,
            )

    return RetrainRunResult(
        modality="image", ran=True, promoted=promoted, metrics=metrics, version_tag=version_tag
    )


def _promote_checkpoint(candidate_dir: Path, live_dir: Path) -> None:
    """
    Copy a newly promoted checkpoint into the "live" directory that
    ``TextModerationModel.from_pretrained`` / ``ImageModerationModel.from_pretrained``
    read from by default, so the promotion takes effect without needing to
    change ``settings`` at runtime.
    """
    import shutil

    live_dir.mkdir(parents=True, exist_ok=True)
    for item in candidate_dir.iterdir():
        dest = live_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
    logger.info("Promoted checkpoint from %s to live path %s", candidate_dir, live_dir)


def run(
    text_data_path: Path = Path("data/text_moderation_dataset.csv"),
    skip_text: bool = False,
    skip_image: bool = False,
) -> list[RetrainRunResult]:
    """Run the full weekly retraining cycle. Returns per-modality results."""
    logger.info("=== Starting weekly retraining pipeline run ===")
    results: list[RetrainRunResult] = []

    if not skip_text:
        try:
            results.append(run_text_retraining(text_data_path, settings.retrain_min_new_samples))
        except Exception:
            logger.exception("Text retraining failed")
            results.append(RetrainRunResult(modality="text", ran=False, reason="exception during training"))
    else:
        results.append(RetrainRunResult(modality="text", ran=False, reason="skipped by flag"))

    if not skip_image:
        try:
            results.append(run_image_retraining(settings.retrain_min_new_samples))
        except Exception:
            logger.exception("Image retraining failed")
            results.append(RetrainRunResult(modality="image", ran=False, reason="exception during training"))
    else:
        results.append(RetrainRunResult(modality="image", ran=False, reason="skipped by flag"))

    with session_scope() as db:
        db.add(
            AuditLog(
                moderation_request_id=None,
                event_type="retrain_pipeline_completed",
                actor="system",
                details={"results": [r.__dict__ for r in results]},
            )
        )

    logger.info("=== Retraining pipeline run complete: %s ===", [r.__dict__ for r in results])
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the weekly content-moderation retraining pipeline.")
    parser.add_argument(
        "--text-data-path", type=Path, default=Path("data/text_moderation_dataset.csv")
    )
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(text_data_path=args.text_data_path, skip_text=args.skip_text, skip_image=args.skip_image)
