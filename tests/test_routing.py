"""Tests for app.services.routing — the confidence-based routing engine (Module 4)."""

from __future__ import annotations

import pytest

from app.schemas.image_moderation import ImageLabelScore, ImageModerationResult
from app.schemas.routing import RoutingAction, RoutingTier
from app.schemas.text_moderation import LabelScore, TextModerationResult
from app.services.routing import ConfidenceRouter

TEXT_LABELS = ["toxic", "spam", "hate"]
IMAGE_LABELS = ["nsfw", "violence", "weapons", "hate_symbols", "graphic_content"]


def make_text_result(top_label: str, top_score: float, flagged: list[str]) -> TextModerationResult:
    labels = [
        LabelScore(label=l, score=(top_score if l == top_label else 0.05), predicted=(l in flagged))
        for l in TEXT_LABELS
    ]
    return TextModerationResult(
        text="sample", labels=labels, predicted_labels=flagged, top_label=top_label,
        top_score=top_score, model_version="v1",
    )


def make_image_result(top_label: str, top_score: float, flagged: list[str]) -> ImageModerationResult:
    labels = [
        ImageLabelScore(label=l, score=(top_score if l == top_label else 0.05), predicted=(l in flagged))
        for l in IMAGE_LABELS
    ]
    return ImageModerationResult(
        source="img", labels=labels, predicted_labels=flagged, top_label=top_label,
        top_score=top_score, model_version="clip-v1",
    )


@pytest.fixture()
def router() -> ConfidenceRouter:
    return ConfidenceRouter(auto_threshold=0.95, queue_threshold=0.70)


@pytest.mark.parametrize(
    "top_label,top_score,flagged,expected_tier,expected_action",
    [
        # No labels flagged → model says "clean" → auto-approve (regardless of score)
        ("toxic", 0.98, [], RoutingTier.AUTO, RoutingAction.AUTO_APPROVE),
        ("nsfw", 0.55, [], RoutingTier.AUTO, RoutingAction.AUTO_APPROVE),
        # Labels flagged → use confidence thresholds
        ("toxic", 0.97, ["toxic"], RoutingTier.AUTO, RoutingAction.AUTO_REJECT),
        ("spam", 0.80, ["spam"], RoutingTier.QUEUE, RoutingAction.MODERATOR_QUEUE),
        # Boundary: exactly at auto_threshold must NOT be "auto" (spec says "> 95%", strict).
        ("toxic", 0.95, ["toxic"], RoutingTier.QUEUE, RoutingAction.MODERATOR_QUEUE),
        # Boundary: exactly at queue_threshold IS "queue" (inclusive per spec).
        ("spam", 0.70, ["spam"], RoutingTier.QUEUE, RoutingAction.MODERATOR_QUEUE),
        # Low confidence with flagged labels → UNCERTAIN
        ("nsfw", 0.55, ["nsfw"], RoutingTier.UNCERTAIN, RoutingAction.MANUAL_REVIEW_UNCERTAIN),
    ],
)
def test_route_text_tiers(router, top_label, top_score, flagged, expected_tier, expected_action):
    result = make_text_result(top_label, top_score, flagged)
    decision = router.route_text(result)
    assert decision.tier == expected_tier
    assert decision.action == expected_action
    assert decision.confidence == top_score
    assert decision.source == "text"


def test_route_image_mirrors_text_policy(router):
    result = make_image_result("weapons", 0.99, ["weapons"])
    decision = router.route_image(result)
    assert decision.tier == RoutingTier.AUTO
    assert decision.action == RoutingAction.AUTO_REJECT
    assert decision.flagged_labels == ["image:weapons"]


def test_flagged_labels_are_tagged_by_modality(router):
    text_decision = router.route_text(make_text_result("toxic", 0.97, ["toxic"]))
    assert text_decision.flagged_labels == ["text:toxic"]


class TestCombinedRouting:
    def test_reject_wins_over_approve(self, router):
        text = make_text_result("toxic", 0.99, [])
        image = make_image_result("weapons", 0.99, ["weapons"])
        decision = router.route_content(text_result=text, image_result=image)
        assert decision.action == RoutingAction.AUTO_REJECT
        assert decision.source == "combined"

    def test_uncertain_beats_queue(self, router):
        # Image with low confidence but no labels flagged → auto-approve (clean)
        # Text with flagged label in queue band → moderator_queue
        # Combined takes the most severe: moderator_queue
        text = make_text_result("spam", 0.80, ["spam"])  # queue
        image = make_image_result("nsfw", 0.50, [])  # auto-approve (no labels flagged)
        decision = router.route_content(text_result=text, image_result=image)
        assert decision.action == RoutingAction.MODERATOR_QUEUE

    def test_both_approve_combined_approves(self, router):
        text = make_text_result("toxic", 0.99, [])
        image = make_image_result("nsfw", 0.99, [])
        decision = router.route_content(text_result=text, image_result=image)
        assert decision.action == RoutingAction.AUTO_APPROVE

    def test_single_modality_still_works(self, router):
        image = make_image_result("weapons", 0.99, ["weapons"])
        decision = router.route_content(image_result=image)
        assert decision.source == "combined"
        assert decision.action == RoutingAction.AUTO_REJECT

    def test_requires_at_least_one_modality(self, router):
        with pytest.raises(ValueError):
            router.route_content()

    def test_combined_flagged_labels_include_both_modalities(self, router):
        text = make_text_result("toxic", 0.97, ["toxic"])
        image = make_image_result("weapons", 0.97, ["weapons"])
        decision = router.route_content(text_result=text, image_result=image)
        assert "text:toxic" in decision.flagged_labels
        assert "image:weapons" in decision.flagged_labels

    def test_combined_model_versions_include_both(self, router):
        text = make_text_result("toxic", 0.99, [])
        image = make_image_result("nsfw", 0.99, [])
        decision = router.route_content(text_result=text, image_result=image)
        assert decision.model_versions == {"text": "v1", "image": "clip-v1"}


def test_requires_human_review_property():
    from app.schemas.routing import RoutingDecision

    auto = RoutingDecision(
        source="text", tier=RoutingTier.AUTO, action=RoutingAction.AUTO_APPROVE,
        confidence=0.99, reasoning="x",
    )
    queued = RoutingDecision(
        source="text", tier=RoutingTier.QUEUE, action=RoutingAction.MODERATOR_QUEUE,
        confidence=0.8, reasoning="x",
    )
    assert auto.requires_human_review is False
    assert queued.requires_human_review is True
