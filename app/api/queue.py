"""
GET /moderation_queue, POST /review

The human-in-the-loop surface: list items awaiting a moderator, and submit
a moderator's decision on one. Built directly on Module 5's queue service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.logging_config import get_logger
from app.db.models import Modality, RequestStatus, ReviewDecision
from app.schemas.api import QueueItem, QueuePredictionSummary, QueueResponse, ReviewRequest, ReviewResponse
from app.services.queue_service import list_queue, submit_review

logger = get_logger(__name__)
router = APIRouter(tags=["human-in-the-loop"])

_STATUS_ALIASES = {
    "queued": RequestStatus.QUEUED,
    "uncertain": RequestStatus.UNCERTAIN,
    "pending": RequestStatus.PENDING,
    "auto_approved": RequestStatus.AUTO_APPROVED,
    "auto_rejected": RequestStatus.AUTO_REJECTED,
    "reviewed": RequestStatus.REVIEWED,
}


@router.get("/moderation_queue", response_model=QueueResponse)
def get_moderation_queue(
    status: list[str] | None = Query(
        default=None, description="Filter by status; defaults to queued+uncertain."
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> QueueResponse:
    """List moderation requests awaiting human review."""
    statuses = None
    if status:
        try:
            statuses = [_STATUS_ALIASES[s.lower()] for s in status]
        except KeyError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid status value: {exc}. Allowed: {list(_STATUS_ALIASES)}"
            ) from exc

    requests = list_queue(db, statuses=statuses, limit=limit, offset=offset)

    items = [
        QueueItem(
            id=req.id,
            content_type=req.content_type.value,
            text_content=req.text_content,
            image_path=req.image_path,
            status=req.status.value,
            created_at=req.created_at,
            predictions=[
                QueuePredictionSummary(
                    modality=p.modality.value,
                    model_version=p.model_version,
                    top_label=p.top_label,
                    top_score=p.top_score,
                    predicted_labels=p.predicted_labels,
                )
                for p in req.predictions
            ],
        )
        for req in requests
    ]

    return QueueResponse(items=items, count=len(items), limit=limit, offset=offset)


@router.post("/review", response_model=ReviewResponse)
def review_moderation_request(
    payload: ReviewRequest,
    db: Session = Depends(get_db),
) -> ReviewResponse:
    """Submit a moderator's decision on a queued or uncertain request."""
    try:
        review = submit_review(
            db,
            moderation_request_id=payload.moderation_request_id,
            decision=ReviewDecision(payload.decision),
            moderator_id=payload.moderator_id,
            notes=payload.notes,
            corrected_labels=payload.corrected_labels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ReviewResponse(
        review_id=review.id,
        moderation_request_id=review.moderation_request_id,
        decision=review.decision.value,
        reviewed_at=review.reviewed_at,
    )
