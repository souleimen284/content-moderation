"""
Pydantic schemas for image moderation requests and results.

Mirrors ``app.schemas.text_moderation`` so the API layer (Module 7) can
expose a consistent response shape across ``POST /moderate/text``,
``POST /moderate/image``, and the combined ``POST /moderate/content``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageLabelScore(BaseModel):
    """Model confidence that an image belongs to a single moderation category."""

    label: str = Field(..., description="Category name, e.g. 'nsfw'.")
    score: float = Field(..., ge=0.0, le=1.0, description="Zero-shot match probability in [0, 1].")
    predicted: bool = Field(..., description="Whether score crosses the per-label decision cutoff.")


class ImageModerationResult(BaseModel):
    """Full multi-label prediction output for one image."""

    source: str = Field(default="upload", description="Filename or identifier of the image.")
    labels: list[ImageLabelScore]
    predicted_labels: list[str] = Field(
        default_factory=list, description="Labels where `predicted` is True."
    )
    top_label: str = Field(..., description="Highest-confidence category.")
    top_score: float = Field(..., ge=0.0, le=1.0, description="Confidence of top_label.")
    model_version: str = Field(default="unversioned")
    decision: str = Field(default="REVIEW", description="SAFE, REVIEW, or NSFW.")
    nsfw_score: float = Field(default=0.0, ge=0.0, le=1.0)
    method: str = Field(default="supervised-pretrained-inference")
    categories: dict[str, float] = Field(default_factory=dict)

    # --- New fields for enhanced NSFW detection ---
    nsfw_score: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description=(
            "NSFW pseudo-probability from prompt ensemble softmax. "
            "This is NOT a true calibrated probability — it represents the "
            "relative confidence that the image is NSFW vs SAFE based on "
            "average similarity to NSFW and SAFE prompt ensembles."
        ),
    )
    decision: str = Field(
        default="SAFE",
        description="Three-way routing decision: SAFE, REVIEW, or NSFW.",
    )
    method: str = Field(
        default="zero-shot",
        description="Classification method used (e.g. 'zero-shot-clip-prompt-ensemble').",
    )

    @property
    def is_flagged(self) -> bool:
        return len(self.predicted_labels) > 0
