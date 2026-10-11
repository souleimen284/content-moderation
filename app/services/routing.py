"""
Confidence-based routing engine.

Implements the three-tier moderation policy:

  - confidence > auto_decision_threshold  -> AUTO tier -> auto-approve or
    auto-reject, decided by whether any category was actually flagged.
  - queue_threshold <= confidence <= auto_decision_threshold -> QUEUE tier
    -> sent to the moderator queue.
  - confidence < queue_threshold -> UNCERTAIN tier -> marked uncertain for
    manual review.

All thresholds are configurable via ``app.core.config.settings`` (env vars
``AUTO_DECISION_THRESHOLD`` / ``QUEUE_THRESHOLD``), so this module contains
no hardcoded cutoffs — it only contains the *policy* of what to do once a
tier has been determined.

This is the single place where text and image predictions get converted
into an actual moderation action, so both ``POST /moderate/text`` and
``POST /moderate/image`` (Module 7) call into this router rather than each
re-implementing threshold logic.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.image_moderation import ImageModerationResult
from app.schemas.routing import ACTION_SEVERITY, RoutingAction, RoutingDecision, RoutingTier
from app.schemas.text_moderation import TextModerationResult

logger = get_logger(__name__)


class ConfidenceRouter:
    """Applies the configurable confidence-routing policy to model outputs."""

    def __init__(self, auto_threshold: float | None = None, queue_threshold: float | None = None) -> None:
        # Defaults come from global settings, but can be overridden per
        # instance (e.g. per-tenant policies, or in tests).
        self.auto_threshold = auto_threshold if auto_threshold is not None else settings.auto_decision_threshold
        self.queue_threshold = queue_threshold if queue_threshold is not None else settings.queue_threshold

    def _tier_for(self, confidence: float, is_flagged: bool = False) -> RoutingTier:
        # If no labels are flagged the model is saying "this is clean" —
        # auto-approve regardless of the raw top_score.  Without this,
        # clean content always lands in UNCERTAIN because its top_score
        # (the max probability across all moderation labels) is naturally
        # low, causing 60 %+ of traffic to queue for human review.
        if not is_flagged:
            return RoutingTier.AUTO
        if confidence > self.auto_threshold:
            return RoutingTier.AUTO
        if confidence >= self.queue_threshold:
            return RoutingTier.QUEUE
        return RoutingTier.UNCERTAIN

    def _action_for(self, tier: RoutingTier, is_flagged: bool) -> RoutingAction:
        if tier == RoutingTier.AUTO:
            return RoutingAction.AUTO_REJECT if is_flagged else RoutingAction.AUTO_APPROVE
        if tier == RoutingTier.QUEUE:
            return RoutingAction.MODERATOR_QUEUE
        return RoutingAction.MANUAL_REVIEW_UNCERTAIN

    # ------------------------------------------------------------------ #
    # Single-modality routing
    # ------------------------------------------------------------------ #
    def route_text(self, result: TextModerationResult) -> RoutingDecision:
        """Route a single text moderation result to an action."""
        tier = self._tier_for(result.top_score, result.is_flagged)
        action = self._action_for(tier, result.is_flagged)
        reasoning = self._build_reasoning(
            modality="text",
            tier=tier,
            action=action,
            top_label=result.top_label,
            top_score=result.top_score,
            flagged=result.predicted_labels,
        )
        decision = RoutingDecision(
            source="text",
            tier=tier,
            action=action,
            confidence=result.top_score,
            flagged_labels=[f"text:{label}" for label in result.predicted_labels],
            reasoning=reasoning,
            model_versions={"text": result.model_version},
        )
        logger.info(
            "Text routing decision: tier=%s action=%s confidence=%.4f",
            tier.value,
            action.value,
            result.top_score,
        )
        return decision

    def route_image(self, result: ImageModerationResult) -> RoutingDecision:
        """Route a single image moderation result to an action."""
        tier = self._tier_for(result.top_score, result.is_flagged)
        action = self._action_for(tier, result.is_flagged)
        reasoning = self._build_reasoning(
            modality="image",
            tier=tier,
            action=action,
            top_label=result.top_label,
            top_score=result.top_score,
            flagged=result.predicted_labels,
        )
        decision = RoutingDecision(
            source="image",
            tier=tier,
            action=action,
            confidence=result.top_score,
            flagged_labels=[f"image:{label}" for label in result.predicted_labels],
            reasoning=reasoning,
            model_versions={"image": result.model_version},
        )
        logger.info(
            "Image routing decision: tier=%s action=%s confidence=%.4f",
            tier.value,
            action.value,
            result.top_score,
        )
        return decision

    # ------------------------------------------------------------------ #
    # Combined (multi-modal) routing
    # ------------------------------------------------------------------ #
    def route_content(
        self,
        text_result: TextModerationResult | None = None,
        image_result: ImageModerationResult | None = None,
    ) -> RoutingDecision:
        """
        Route a piece of content that may include both text and an image
        (e.g. a post with a caption and a photo) to a single combined action.

        Policy: the most severe individual action wins —
        auto_reject > manual_review_uncertain > moderator_queue > auto_approve.
        This means if either modality independently warrants rejection, the
        combined content is rejected; content only auto-approves if every
        modality present independently auto-approved.
        """
        if text_result is None and image_result is None:
            raise ValueError("route_content requires at least one of text_result or image_result")

        sub_decisions: list[RoutingDecision] = []
        if text_result is not None:
            sub_decisions.append(self.route_text(text_result))
        if image_result is not None:
            sub_decisions.append(self.route_image(image_result))

        if len(sub_decisions) == 1:
            winning = sub_decisions[0]
        else:
            winning = min(sub_decisions, key=lambda d: ACTION_SEVERITY[d.action])

        combined_flagged = [label for d in sub_decisions for label in d.flagged_labels]
        combined_versions: dict[str, str] = {}
        for d in sub_decisions:
            combined_versions.update(d.model_versions)

        parts = [f"{d.source}={d.action.value} (tier={d.tier.value}, conf={d.confidence:.2f})" for d in sub_decisions]
        reasoning = (
            f"Combined decision from {len(sub_decisions)} modality result(s): {'; '.join(parts)}. "
            f"Overall action '{winning.action.value}' selected via highest-severity rule."
        )

        decision = RoutingDecision(
            source="combined",
            tier=winning.tier,
            action=winning.action,
            confidence=winning.confidence,
            flagged_labels=combined_flagged,
            reasoning=reasoning,
            model_versions=combined_versions,
        )
        logger.info(
            "Combined routing decision: action=%s (from %d modalities)",
            winning.action.value,
            len(sub_decisions),
        )
        return decision

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_reasoning(
        modality: str,
        tier: RoutingTier,
        action: RoutingAction,
        top_label: str,
        top_score: float,
        flagged: list[str],
    ) -> str:
        if action == RoutingAction.AUTO_APPROVE:
            return (
                f"{modality.capitalize()} content auto-approved: top prediction '{top_label}' "
                f"scored {top_score:.2%}, above the auto-decision threshold, with no category "
                f"flagged as violating policy."
            )
        if action == RoutingAction.AUTO_REJECT:
            return (
                f"{modality.capitalize()} content auto-rejected: '{top_label}' scored {top_score:.2%} "
                f"(above the auto-decision threshold) and categories {flagged} were flagged."
            )
        if action == RoutingAction.MODERATOR_QUEUE:
            return (
                f"{modality.capitalize()} content sent to moderator queue: top prediction '{top_label}' "
                f"scored {top_score:.2%}, within the moderate-confidence band — a human should confirm."
            )
        return (
            f"{modality.capitalize()} content marked uncertain: top prediction '{top_label}' scored "
            f"only {top_score:.2%}, below the queue threshold — the model isn't confident enough to "
            f"route automatically, so this requires manual review."
        )


_singleton: ConfidenceRouter | None = None


def get_confidence_router() -> ConfidenceRouter:
    """FastAPI-injectable singleton accessor (Module 7)."""
    global _singleton
    if _singleton is None:
        _singleton = ConfidenceRouter()
    return _singleton
