"""
Request/response schemas for the FastAPI layer.

These compose the ML-layer schemas (Modules 2-3) and the routing schema
(Module 4) into the exact JSON shapes returned by each endpoint, plus a
few request bodies (review submission) not needed anywhere else.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.schemas.image_moderation import ImageModerationResult
from app.schemas.routing import RoutingDecision
from app.schemas.text_moderation import TextModerationResult


# --------------------------------------------------------------------------- #
# POST /moderate/text, /moderate/image, /moderate/content
# --------------------------------------------------------------------------- #
class ModerateTextResponse(BaseModel):
    request_id: str
    status: str = Field(..., description="Resulting ModerationRequest status.")
    prediction: TextModerationResult
    routing: RoutingDecision


class ModerateImageResponse(BaseModel):
    request_id: str
    status: str
    prediction: ImageModerationResult
    routing: RoutingDecision


class ModerateContentResponse(BaseModel):
    request_id: str
    status: str
    text_prediction: TextModerationResult | None = None
    image_prediction: ImageModerationResult | None = None
    routing: RoutingDecision


# --------------------------------------------------------------------------- #
# GET /moderation_queue
# --------------------------------------------------------------------------- #
class QueuePredictionSummary(BaseModel):
    modality: str
    model_version: str
    top_label: str
    top_score: float
    predicted_labels: list[str]


class QueueItem(BaseModel):
    id: str
    content_type: str
    text_content: str | None
    image_path: str | None
    status: str
    created_at: datetime
    predictions: list[QueuePredictionSummary]


class QueueResponse(BaseModel):
    items: list[QueueItem]
    count: int
    limit: int
    offset: int


# --------------------------------------------------------------------------- #
# POST /review
# --------------------------------------------------------------------------- #
class ReviewRequest(BaseModel):
    moderation_request_id: str
    decision: str = Field(..., description="'approve', 'reject', or 'escalate'.")
    moderator_id: str | None = None
    notes: str | None = None
    corrected_labels: list[str] | None = Field(
        default=None, description="Ground-truth labels if the moderator is correcting the model."
    )

    @model_validator(mode="after")
    def _validate_decision(self) -> "ReviewRequest":
        allowed = {"approve", "reject", "escalate"}
        if self.decision not in allowed:
            raise ValueError(f"decision must be one of {allowed}, got {self.decision!r}")
        return self


class ReviewResponse(BaseModel):
    review_id: str
    moderation_request_id: str
    decision: str
    reviewed_at: datetime


# --------------------------------------------------------------------------- #
# GET /metrics
# --------------------------------------------------------------------------- #
class ModelVersionSummary(BaseModel):
    version_tag: str
    is_active: bool
    trained_at: datetime
    training_samples: int
    metrics: dict


class MetricsResponse(BaseModel):
    total_requests: int
    requests_by_status: dict[str, int]
    requests_by_content_type: dict[str, int]
    queue_size: int
    reviewed_count: int
    reviews_by_decision: dict[str, int]
    active_text_model: ModelVersionSummary | None
    active_image_model: ModelVersionSummary | None


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    database: str
    app_version: str
    timestamp: datetime
