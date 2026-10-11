"""
Human-in-the-loop moderation queue service.

This is the workflow glue between the ML/routing layers (Modules 2-4) and
persistence (this module's ``app.db.models``):

  1. ``create_moderation_request`` — a new piece of content arrives.
  2. ``record_prediction`` — store each modality's model output.
  3. ``apply_routing_decision`` — store the RoutingDecision's outcome; if it
     requires a human, the request stays queryable via ``list_queue``.
  4. ``submit_review`` — a moderator resolves a queued/uncertain request,
     with their decision and notes captured for retraining.

Every state-changing call also writes an ``AuditLog`` row, so the full
history of any request is reconstructable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.models import (
    AuditLog,
    ContentType,
    ModerationRequest,
    ModeratorReview,
    Modality,
    Prediction,
    RequestStatus,
    ReviewDecision,
)
from app.schemas.image_moderation import ImageModerationResult
from app.schemas.routing import RoutingAction, RoutingDecision
from app.schemas.text_moderation import TextModerationResult

logger = get_logger(__name__)

_ACTION_TO_STATUS: dict[RoutingAction, RequestStatus] = {
    RoutingAction.AUTO_APPROVE: RequestStatus.AUTO_APPROVED,
    RoutingAction.AUTO_REJECT: RequestStatus.AUTO_REJECTED,
    RoutingAction.MODERATOR_QUEUE: RequestStatus.QUEUED,
    RoutingAction.MANUAL_REVIEW_UNCERTAIN: RequestStatus.UNCERTAIN,
}


def _log_event(
    db: Session,
    event_type: str,
    actor: str,
    moderation_request_id: str | None = None,
    details: dict | None = None,
) -> AuditLog:
    entry = AuditLog(
        moderation_request_id=moderation_request_id,
        event_type=event_type,
        actor=actor,
        details=details or {},
    )
    db.add(entry)
    return entry


def create_moderation_request(
    db: Session,
    content_type: ContentType,
    text_content: str | None = None,
    image_path: str | None = None,
    submitted_by_id: str | None = None,
    request_metadata: dict | None = None,
) -> ModerationRequest:
    """Create a new pending moderation request for a piece of content."""
    request = ModerationRequest(
        content_type=content_type,
        text_content=text_content,
        image_path=image_path,
        submitted_by_id=submitted_by_id,
        request_metadata=request_metadata,
        status=RequestStatus.PENDING,
    )
    db.add(request)
    db.flush()  # populate request.id without committing

    _log_event(db, "request_created", actor=submitted_by_id or "system", moderation_request_id=request.id)
    db.commit()
    db.refresh(request)
    logger.info("Created moderation request %s (content_type=%s)", request.id, content_type.value)
    return request


def record_prediction(
    db: Session,
    moderation_request: ModerationRequest,
    modality: Modality,
    result: TextModerationResult | ImageModerationResult,
    routing_tier: str,
    routing_action: str,
) -> Prediction:
    """Persist a single modality's model prediction for a moderation request."""
    prediction = Prediction(
        moderation_request_id=moderation_request.id,
        modality=modality,
        model_version=result.model_version,
        top_label=result.top_label,
        top_score=result.top_score,
        predicted_labels=result.predicted_labels,
        label_scores={ls.label: ls.score for ls in result.labels},
        routing_tier=routing_tier,
        routing_action=routing_action,
    )
    db.add(prediction)
    _log_event(
        db,
        "prediction_recorded",
        actor="system",
        moderation_request_id=moderation_request.id,
        details={"modality": modality.value, "top_label": result.top_label, "top_score": result.top_score},
    )
    db.commit()
    db.refresh(prediction)
    return prediction


def apply_routing_decision(
    db: Session,
    moderation_request: ModerationRequest,
    decision: RoutingDecision,
) -> ModerationRequest:
    """
    Update a moderation request's status based on a RoutingDecision.

    Auto-resolved requests (approve/reject) are marked resolved immediately.
    Queued/uncertain requests remain open until a moderator calls
    ``submit_review``.
    """
    new_status = _ACTION_TO_STATUS[decision.action]
    moderation_request.status = new_status
    if not decision.action.requires_human:
        moderation_request.resolved_at = datetime.now(timezone.utc)

    _log_event(
        db,
        "routing_decision_applied",
        actor="system",
        moderation_request_id=moderation_request.id,
        details={
            "tier": decision.tier.value,
            "action": decision.action.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        },
    )
    db.commit()
    db.refresh(moderation_request)
    logger.info(
        "Applied routing decision to request %s: status=%s", moderation_request.id, new_status.value
    )
    return moderation_request


def list_queue(
    db: Session,
    statuses: list[RequestStatus] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ModerationRequest]:
    """
    List moderation requests awaiting human review.

    Defaults to QUEUED and UNCERTAIN (the two human-review statuses); pass
    an explicit ``statuses`` list to filter differently (e.g. for an admin
    view showing everything).
    """
    statuses = statuses or [RequestStatus.QUEUED, RequestStatus.UNCERTAIN]
    stmt = (
        select(ModerationRequest)
        .where(ModerationRequest.status.in_(statuses))
        .order_by(ModerationRequest.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).all())


def submit_review(
    db: Session,
    moderation_request_id: str,
    decision: ReviewDecision,
    moderator_id: str | None = None,
    notes: str | None = None,
    corrected_labels: list[str] | None = None,
) -> ModeratorReview:
    """
    Record a moderator's decision on a queued/uncertain moderation request.

    ``corrected_labels`` — the moderator's ground-truth label set for the
    content — is what ``export_moderator_reviewed_text_for_retraining``
    (below) turns into new training rows for ``retrain.py`` (Module 6).
    """
    request = db.get(ModerationRequest, moderation_request_id)
    if request is None:
        raise ValueError(f"No moderation request found with id={moderation_request_id}")
    if request.review is not None:
        raise ValueError(f"Moderation request {moderation_request_id} has already been reviewed")

    review = ModeratorReview(
        moderation_request_id=moderation_request_id,
        moderator_id=moderator_id,
        decision=decision,
        corrected_labels=corrected_labels,
        notes=notes,
    )
    db.add(review)

    request.status = RequestStatus.REVIEWED
    request.resolved_at = datetime.now(timezone.utc)

    _log_event(
        db,
        "moderator_review_submitted",
        actor=moderator_id or "unknown_moderator",
        moderation_request_id=moderation_request_id,
        details={"decision": decision.value, "corrected_labels": corrected_labels},
    )
    db.commit()
    db.refresh(review)
    logger.info(
        "Moderator review submitted for request %s: decision=%s", moderation_request_id, decision.value
    )
    return review


def export_moderator_reviewed_text_for_retraining(db: Session, labels: list[str]) -> pd.DataFrame:
    """
    Build a DataFrame of moderator-reviewed TEXT requests in the exact
    schema ``scripts/train_text_model.py`` expects: a ``text`` column plus
    one binary column per label.

    A row's label columns come from the moderator's ``corrected_labels``
    when provided (ground truth), falling back to the original model
    prediction's labels if the moderator didn't supply corrections (e.g. a
    simple "approve, model was right" decision).

    Only REVIEWED text (or combined) requests with a linked review are
    included — this is the exact data ``retrain.py`` merges into the
    training CSV each week.
    """
    stmt = (
        select(ModerationRequest)
        .join(ModeratorReview, ModerationRequest.id == ModeratorReview.moderation_request_id)
        .where(
            ModerationRequest.status == RequestStatus.REVIEWED,
            ModerationRequest.content_type.in_([ContentType.TEXT, ContentType.COMBINED]),
            ModerationRequest.text_content.is_not(None),
        )
    )
    requests = list(db.scalars(stmt).all())

    rows: list[dict] = []
    for req in requests:
        review = req.review
        if review is None:
            continue

        text_prediction = next((p for p in req.predictions if p.modality == Modality.TEXT), None)
        source_labels = review.corrected_labels
        if source_labels is None:
            source_labels = text_prediction.predicted_labels if text_prediction else []

        row = {"text": req.text_content}
        for label in labels:
            row[label] = int(label in source_labels)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["text", *labels])
    logger.info("Exported %d moderator-reviewed text rows for retraining", len(df))
    return df


def export_moderator_reviewed_images_for_retraining(
    db: Session, labels: list[str]
) -> list[dict]:
    """
    Build a list of moderator-reviewed IMAGE requests suitable for training
    the CLIP linear probe (``scripts/train_image_model.py``).

    Each item is ``{"image_path": str, "labels": [binary ints, ...]}`` in
    the same label order as ``labels``, sourced from the moderator's
    ``corrected_labels`` when available (ground truth), falling back to the
    original model prediction otherwise — mirroring the text export.
    """
    stmt = (
        select(ModerationRequest)
        .join(ModeratorReview, ModerationRequest.id == ModeratorReview.moderation_request_id)
        .where(
            ModerationRequest.status == RequestStatus.REVIEWED,
            ModerationRequest.content_type.in_([ContentType.IMAGE, ContentType.COMBINED]),
            ModerationRequest.image_path.is_not(None),
        )
    )
    requests = list(db.scalars(stmt).all())

    rows: list[dict] = []
    for req in requests:
        review = req.review
        if review is None:
            continue

        image_prediction = next((p for p in req.predictions if p.modality == Modality.IMAGE), None)
        source_labels = review.corrected_labels
        if source_labels is None:
            source_labels = image_prediction.predicted_labels if image_prediction else []

        rows.append(
            {
                "image_path": req.image_path,
                "labels": [int(label in source_labels) for label in labels],
            }
        )

    logger.info("Exported %d moderator-reviewed image rows for retraining", len(rows))
    return rows
