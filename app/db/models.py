"""
SQLAlchemy ORM models for the content moderation pipeline.

Six normalized tables, per spec:

  - ``User``              — accounts (submitters, moderators, admins)
  - ``ModerationRequest``  — one row per piece of content submitted for moderation
  - ``Prediction``         — one row per model prediction (text and/or image) for a request
  - ``ModeratorReview``    — one row per human decision on a queued/uncertain request
  - ``ModelVersion``       — registry of trained model checkpoints and their eval metrics
  - ``AuditLog``           — append-only trail of every state-changing event

Relationships: a ``ModerationRequest`` has many ``Prediction`` rows (one per
modality) and at most one ``ModeratorReview``. ``AuditLog`` rows reference a
request but are never updated or deleted, so the moderation history is
always fully reconstructable.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class UserRole(str, enum.Enum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"


class ContentType(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    COMBINED = "combined"


class RequestStatus(str, enum.Enum):
    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    AUTO_REJECTED = "auto_rejected"
    QUEUED = "queued"
    UNCERTAIN = "uncertain"
    REVIEWED = "reviewed"


class ReviewDecision(str, enum.Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class Modality(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    moderation_requests: Mapped[list["ModerationRequest"]] = relationship(
        back_populates="submitted_by_user", foreign_keys="ModerationRequest.submitted_by_id"
    )
    reviews: Mapped[list["ModeratorReview"]] = relationship(back_populates="moderator")

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return f"<User id={self.id} username={self.username!r} role={self.role.value}>"


class ModerationRequest(Base):
    __tablename__ = "moderation_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType), nullable=False)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[RequestStatus] = mapped_column(
        Enum(RequestStatus), default=RequestStatus.PENDING, nullable=False, index=True
    )
    submitted_by_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True
    )
    request_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    submitted_by_user: Mapped["User | None"] = relationship(
        back_populates="moderation_requests", foreign_keys=[submitted_by_id]
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="moderation_request", cascade="all, delete-orphan"
    )
    review: Mapped["ModeratorReview | None"] = relationship(
        back_populates="moderation_request", uselist=False, cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        back_populates="moderation_request", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ModerationRequest id={self.id} status={self.status.value}>"


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    moderation_request_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("moderation_requests.id"), nullable=False, index=True
    )
    modality: Mapped[Modality] = mapped_column(Enum(Modality), nullable=False)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    top_label: Mapped[str] = mapped_column(String(100), nullable=False)
    top_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_labels: Mapped[list] = mapped_column(JSON, default=list)
    label_scores: Mapped[dict] = mapped_column(JSON, default=dict)  # {label: score, ...}
    routing_tier: Mapped[str] = mapped_column(String(20), nullable=False)
    routing_action: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    moderation_request: Mapped["ModerationRequest"] = relationship(back_populates="predictions")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Prediction id={self.id} modality={self.modality.value} top_label={self.top_label}>"


class ModeratorReview(Base):
    __tablename__ = "moderator_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    moderation_request_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("moderation_requests.id"), unique=True, nullable=False
    )
    moderator_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    decision: Mapped[ReviewDecision] = mapped_column(Enum(ReviewDecision), nullable=False)
    corrected_labels: Mapped[list | None] = mapped_column(
        JSON, nullable=True, doc="Moderator's corrected label set, used for retraining."
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    moderation_request: Mapped["ModerationRequest"] = relationship(back_populates="review")
    moderator: Mapped["User | None"] = relationship(back_populates="reviews")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ModeratorReview id={self.id} decision={self.decision.value}>"


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    model_type: Mapped[Modality] = mapped_column(Enum(Modality), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(100), nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    training_samples: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ModelVersion type={self.model_type.value} tag={self.version_tag} active={self.is_active}>"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    moderation_request_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("moderation_requests.id"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    actor: Mapped[str] = mapped_column(
        String(150), nullable=False, doc="'system' for automated events, or a user id."
    )
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    moderation_request: Mapped["ModerationRequest | None"] = relationship(back_populates="audit_logs")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AuditLog event={self.event_type} actor={self.actor}>"
