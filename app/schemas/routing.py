"""
Pydantic schemas for the confidence-based routing engine.

A ``RoutingTier`` describes *how confident* the model was (mirrors
``Settings.classify_confidence``'s three-way output). A ``RoutingAction``
describes *what actually happens* to the content — these are different
things: a high-confidence prediction can result in either an approval or a
rejection depending on whether the content was flagged, and only the
`auto` tier resolves directly into an automatic approve/reject; `queue`
and `uncertain` both hand off to a human, differing only in how urgently /
how the item should be triaged.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RoutingTier(str, Enum):
    """Confidence band, matching Settings.classify_confidence()."""

    AUTO = "auto"          # confidence > auto_decision_threshold
    QUEUE = "queue"        # queue_threshold <= confidence <= auto_decision_threshold
    UNCERTAIN = "uncertain"  # confidence < queue_threshold


class RoutingAction(str, Enum):
    """The actual disposition applied to the content."""

    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    MODERATOR_QUEUE = "moderator_queue"
    MANUAL_REVIEW_UNCERTAIN = "manual_review_uncertain"

    @property
    def requires_human(self) -> bool:
        return self in (RoutingAction.MODERATOR_QUEUE, RoutingAction.MANUAL_REVIEW_UNCERTAIN)


#: Severity ordering used to combine multi-modality decisions: lower rank
#: wins when merging a text decision and an image decision into one.
ACTION_SEVERITY: dict[RoutingAction, int] = {
    RoutingAction.AUTO_REJECT: 0,
    RoutingAction.MANUAL_REVIEW_UNCERTAIN: 1,
    RoutingAction.MODERATOR_QUEUE: 2,
    RoutingAction.AUTO_APPROVE: 3,
}


class RoutingDecision(BaseModel):
    """The output of the routing engine for one moderation request."""

    source: str = Field(..., description="'text', 'image', or 'combined'.")
    tier: RoutingTier
    action: RoutingAction
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence driving this decision.")
    flagged_labels: list[str] = Field(
        default_factory=list, description="Categories predicted positive, tagged by modality."
    )
    reasoning: str = Field(..., description="Human-readable explanation for audit/queue display.")
    model_versions: dict[str, str] = Field(
        default_factory=dict, description="e.g. {'text': 'v3', 'image': 'ViT-B-32:openai'}"
    )

    @property
    def requires_human_review(self) -> bool:
        return self.action.requires_human
