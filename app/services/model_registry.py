"""
Model version registry.

Wraps ``app.db.models.ModelVersion`` with the promotion policy used by the
weekly retraining pipeline (``retrain.py``): a freshly retrained model is
only promoted to "active" — i.e. the one ``get_text_moderation_model()`` /
``get_image_moderation_model()`` will load next — if it beats the
currently active model on a chosen metric. Otherwise the previous model
stays active and the attempt is recorded for visibility, not silently
discarded.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.models import AuditLog, Modality, ModelVersion

logger = get_logger(__name__)


def get_active_model_version(db: Session, model_type: Modality) -> ModelVersion | None:
    """Return the currently active ModelVersion for a modality, if any."""
    stmt = select(ModelVersion).where(
        ModelVersion.model_type == model_type, ModelVersion.is_active.is_(True)
    )
    return db.scalars(stmt).first()


def register_model_version(
    db: Session,
    model_type: Modality,
    version_tag: str,
    checkpoint_path: str,
    metrics: dict,
    training_samples: int,
    activate: bool = False,
) -> ModelVersion:
    """
    Insert a new ModelVersion row. If ``activate`` is True, deactivates any
    previously active version of the same modality first, so exactly one
    version per modality is active at a time.
    """
    if activate:
        _deactivate_all(db, model_type)

    version = ModelVersion(
        model_type=model_type,
        version_tag=version_tag,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        training_samples=training_samples,
        is_active=activate,
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    logger.info(
        "Registered model version %s (%s) active=%s", version_tag, model_type.value, activate
    )
    return version


def _score(metrics: dict, metric_key: str) -> float | None:
    """Look up a metric that may be stored with the ``eval_`` prefix.

    ``trainer.evaluate()`` returns keys like ``eval_f1_macro``, while
    callers pass the bare name (``f1_macro``). Accept both so the
    candidate-vs-active comparison actually compares like for like.
    """
    if metric_key in metrics:
        return metrics[metric_key]
    return metrics.get(f"eval_{metric_key}")


def promote_if_better(
    db: Session,
    model_type: Modality,
    candidate_version_tag: str,
    candidate_checkpoint_path: str,
    candidate_metrics: dict,
    training_samples: int,
    metric_key: str = "f1_macro",
) -> tuple[ModelVersion, bool]:
    """
    Register a newly trained candidate model and decide whether to promote
    it to active, based on comparing ``metric_key`` against the currently
    active model for this modality.

    Returns:
        (the ModelVersion that ends up active, whether the candidate was promoted)
    """
    current = get_active_model_version(db, model_type)
    candidate_score = _score(candidate_metrics, metric_key)

    if current is None:
        logger.info(
            "No active %s model yet — promoting candidate %s unconditionally.",
            model_type.value,
            candidate_version_tag,
        )
        promoted = True
    else:
        current_score = _score(current.metrics, metric_key) or float("-inf")
        promoted = candidate_score is not None and candidate_score > current_score
        logger.info(
            "Comparing candidate %s (%s=%.4f) vs active %s (%s=%.4f) -> promote=%s",
            candidate_version_tag,
            metric_key,
            candidate_score if candidate_score is not None else float("nan"),
            current.version_tag,
            metric_key,
            current_score,
            promoted,
        )

    new_version = register_model_version(
        db,
        model_type=model_type,
        version_tag=candidate_version_tag,
        checkpoint_path=candidate_checkpoint_path,
        metrics=candidate_metrics,
        training_samples=training_samples,
        activate=promoted,
    )

    db.add(
        AuditLog(
            moderation_request_id=None,
            event_type="model_retrain_evaluated",
            actor="system",
            details={
                "model_type": model_type.value,
                "candidate_version": candidate_version_tag,
                "candidate_metric": candidate_score,
                "metric_key": metric_key,
                "promoted": promoted,
                "previous_active_version": current.version_tag if current else None,
            },
        )
    )
    db.commit()

    active_version = new_version if promoted else current
    if active_version is None:
        # Only reachable if promotion failed AND no prior active model existed,
        # which shouldn't happen since we promote unconditionally in that case.
        active_version = new_version
    return active_version, promoted


def _deactivate_all(db: Session, model_type: Modality) -> None:
    stmt = select(ModelVersion).where(
        ModelVersion.model_type == model_type, ModelVersion.is_active.is_(True)
    )
    for version in db.scalars(stmt).all():
        version.is_active = False
    db.flush()
