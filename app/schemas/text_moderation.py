"""
Pydantic schemas for text moderation requests and results.

These models are the contract between the ML layer (``app.ml.text_moderation``),
the confidence-routing service (Module 4), and the FastAPI endpoints
(``POST /moderate/text``, Module 7). Keeping them here — rather than defined
inline in the API router — lets the ML layer stay framework-agnostic and
importable from the retraining script, tests, and the dashboard alike.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TextModerationRequest(BaseModel):
    """Input payload for a single piece of text to moderate."""

    text: str = Field(..., min_length=1, max_length=10_000, description="Raw text to moderate.")
    metadata: dict | None = Field(
        default=None, description="Optional caller-supplied metadata (user_id, source, etc.)."
    )

    @field_validator("text")
    @classmethod
    def _strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace only")
        return stripped


class LabelScore(BaseModel):
    """Model confidence for a single moderation category."""

    label: str = Field(..., description="Category name, e.g. 'toxic'.")
    score: float = Field(..., ge=0.0, le=1.0, description="Sigmoid probability in [0, 1].")
    predicted: bool = Field(..., description="Whether score crosses the per-label decision cutoff.")


class TextModerationResult(BaseModel):
    """Full multi-label prediction output for one piece of text."""

    text: str
    labels: list[LabelScore]
    predicted_labels: list[str] = Field(
        default_factory=list, description="Labels where `predicted` is True."
    )
    top_label: str = Field(..., description="Highest-confidence category.")
    top_score: float = Field(..., ge=0.0, le=1.0, description="Confidence of top_label.")
    model_version: str = Field(default="unversioned")

    @property
    def is_flagged(self) -> bool:
        """True if any category other than a clean/benign placeholder was predicted."""
        return len(self.predicted_labels) > 0
