"""
GET /metrics, GET /health

Operational endpoints: live aggregate stats over the moderation database,
and a basic health/readiness check for load balancers and orchestrators.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.api.deps import get_db
from app.core.logging_config import get_logger
from app.db.models import ModelVersion
from app.schemas.api import HealthResponse, MetricsResponse, ModelVersionSummary
from app.services.analytics_service import get_overall_metrics

logger = get_logger(__name__)
router = APIRouter(tags=["system"])


def _to_summary(version: ModelVersion | None) -> ModelVersionSummary | None:
    if version is None:
        return None
    return ModelVersionSummary(
        version_tag=version.version_tag,
        is_active=version.is_active,
        trained_at=version.trained_at,
        training_samples=version.training_samples,
        metrics=version.metrics,
    )


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(db: Session = Depends(get_db)) -> MetricsResponse:
    """Live aggregate moderation statistics, computed via the shared analytics service."""
    stats = get_overall_metrics(db)
    return MetricsResponse(
        total_requests=stats["total_requests"],
        requests_by_status=stats["requests_by_status"],
        requests_by_content_type=stats["requests_by_content_type"],
        queue_size=stats["queue_size"],
        reviewed_count=stats["reviewed_count"],
        reviews_by_decision=stats["reviews_by_decision"],
        active_text_model=_to_summary(stats["active_text_model"]),
        active_image_model=_to_summary(stats["active_image_model"]),
    )


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Basic liveness/readiness check: confirms the API can reach the database."""
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        logger.exception("Health check database connectivity failure")
        db_status = "error"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        database=db_status,
        app_version=__version__,
        timestamp=datetime.now(timezone.utc),
    )
