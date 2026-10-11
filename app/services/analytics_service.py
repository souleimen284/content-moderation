"""
Shared analytics/reporting queries.

Factored out so ``app.api.system.get_metrics`` (Module 7) and the
Streamlit dashboard (Module 8) compute every statistic identically,
against the same database, rather than maintaining two versions of the
same aggregation logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.models import (
    Modality,
    ModerationRequest,
    ModeratorReview,
    ModelVersion,
    Prediction,
)
from app.services.model_registry import get_active_model_version

logger = get_logger(__name__)


def get_overall_metrics(db: Session) -> dict:
    """The same aggregate stats returned by GET /metrics."""
    total_requests = db.scalar(select(func.count(ModerationRequest.id))) or 0

    status_rows = db.execute(
        select(ModerationRequest.status, func.count(ModerationRequest.id)).group_by(
            ModerationRequest.status
        )
    ).all()
    requests_by_status = {status.value: count for status, count in status_rows}

    content_type_rows = db.execute(
        select(ModerationRequest.content_type, func.count(ModerationRequest.id)).group_by(
            ModerationRequest.content_type
        )
    ).all()
    requests_by_content_type = {ct.value: count for ct, count in content_type_rows}

    queue_size = requests_by_status.get("queued", 0) + requests_by_status.get("uncertain", 0)
    reviewed_count = db.scalar(select(func.count(ModeratorReview.id))) or 0

    review_decision_rows = db.execute(
        select(ModeratorReview.decision, func.count(ModeratorReview.id)).group_by(
            ModeratorReview.decision
        )
    ).all()
    reviews_by_decision = {decision.value: count for decision, count in review_decision_rows}

    return {
        "total_requests": total_requests,
        "requests_by_status": requests_by_status,
        "requests_by_content_type": requests_by_content_type,
        "queue_size": queue_size,
        "reviewed_count": reviewed_count,
        "reviews_by_decision": reviews_by_decision,
        "active_text_model": get_active_model_version(db, Modality.TEXT),
        "active_image_model": get_active_model_version(db, Modality.IMAGE),
    }


def get_daily_volume(db: Session, days: int = 30) -> pd.DataFrame:
    """Moderation request volume per day, for the last ``days`` days."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        select(ModerationRequest.created_at, ModerationRequest.status).where(
            ModerationRequest.created_at >= since
        )
    ).all()

    if not rows:
        return pd.DataFrame(columns=["date", "count"])

    df = pd.DataFrame(rows, columns=["created_at", "status"])
    df["date"] = pd.to_datetime(df["created_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    return daily.sort_values("date")


def get_confidence_scores(db: Session, modality: Modality | None = None) -> pd.DataFrame:
    """Raw top_score values for every prediction, for histogramming."""
    stmt = select(Prediction.modality, Prediction.top_score)
    if modality is not None:
        stmt = stmt.where(Prediction.modality == modality)
    rows = db.execute(stmt).all()
    return pd.DataFrame(rows, columns=["modality", "top_score"])


def get_category_distribution(db: Session, modality: Modality | None = None) -> pd.DataFrame:
    """Frequency of each flagged label across all predictions."""
    stmt = select(Prediction.modality, Prediction.predicted_labels)
    if modality is not None:
        stmt = stmt.where(Prediction.modality == modality)
    rows = db.execute(stmt).all()

    counts: dict[str, int] = {}
    for _, labels in rows:
        for label in labels or []:
            counts[label] = counts.get(label, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["label", "count"])

    df = pd.DataFrame(list(counts.items()), columns=["label", "count"])
    return df.sort_values("count", ascending=False)


def get_moderator_performance(db: Session) -> pd.DataFrame:
    """Per-moderator review counts, decision breakdown, and average review latency."""
    stmt = select(
        ModeratorReview.moderator_id,
        ModeratorReview.decision,
        ModeratorReview.reviewed_at,
        ModerationRequest.created_at,
    ).join(ModerationRequest, ModeratorReview.moderation_request_id == ModerationRequest.id)
    rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame(
            columns=["moderator_id", "total_reviews", "approve", "reject", "escalate", "avg_review_seconds"]
        )

    df = pd.DataFrame(rows, columns=["moderator_id", "decision", "reviewed_at", "request_created_at"])
    df["moderator_id"] = df["moderator_id"].fillna("unassigned")
    df["decision"] = df["decision"].apply(lambda d: d.value if hasattr(d, "value") else d)
    df["review_seconds"] = (
        pd.to_datetime(df["reviewed_at"]) - pd.to_datetime(df["request_created_at"])
    ).dt.total_seconds()

    summary = (
        df.groupby("moderator_id")
        .agg(total_reviews=("decision", "count"), avg_review_seconds=("review_seconds", "mean"))
        .reset_index()
    )
    decision_pivot = pd.crosstab(df["moderator_id"], df["decision"]).reset_index()

    result = summary.merge(decision_pivot, on="moderator_id", how="left")
    for col in ("approve", "reject", "escalate"):
        if col not in result.columns:
            result[col] = 0
    return result


def get_active_model_confusion_matrices(db: Session) -> dict[str, dict | None]:
    """Confusion matrices from each active model's stored training metrics."""
    text_version = get_active_model_version(db, Modality.TEXT)
    image_version = get_active_model_version(db, Modality.IMAGE)
    return {
        "text": (text_version.metrics or {}).get("confusion_matrix") if text_version else None,
        "image": (image_version.metrics or {}).get("confusion_matrix") if image_version else None,
    }


def get_all_model_versions(db: Session, model_type: Modality) -> list[ModelVersion]:
    """Full version history for a modality, most recent first (for a timeline view)."""
    stmt = (
        select(ModelVersion)
        .where(ModelVersion.model_type == model_type)
        .order_by(ModelVersion.trained_at.desc())
    )
    return list(db.scalars(stmt).all())
