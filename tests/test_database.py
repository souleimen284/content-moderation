"""Tests for app.db.models and app.services.queue_service (Module 5)."""

from __future__ import annotations

import pytest

from app.db.models import ContentType, Modality, RequestStatus, ReviewDecision
from app.schemas.text_moderation import LabelScore, TextModerationResult
from app.services.queue_service import (
    apply_routing_decision,
    create_moderation_request,
    export_moderator_reviewed_text_for_retraining,
    list_queue,
    record_prediction,
    submit_review,
)
from app.services.routing import ConfidenceRouter

TEXT_LABELS = ["toxic", "spam", "hate"]


def make_text_result(text, top_label, top_score, flagged):
    labels = [
        LabelScore(label=l, score=(top_score if l == top_label else 0.05), predicted=(l in flagged))
        for l in TEXT_LABELS
    ]
    return TextModerationResult(
        text=text, labels=labels, predicted_labels=flagged, top_label=top_label,
        top_score=top_score, model_version="bert-v1",
    )


@pytest.fixture()
def router():
    return ConfidenceRouter(auto_threshold=0.95, queue_threshold=0.70)


def test_create_moderation_request(db_session):
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="hello")
    assert req.id is not None
    assert req.status == RequestStatus.PENDING
    assert req.text_content == "hello"


def test_full_lifecycle_mid_confidence_gets_queued(db_session, router):
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="free money now")
    result = make_text_result("free money now", "spam", 0.80, ["spam"])
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    updated = apply_routing_decision(db_session, req, decision)

    assert updated.status == RequestStatus.QUEUED
    assert updated.resolved_at is None  # not resolved until a human reviews it


def test_high_confidence_clean_auto_approves(db_session, router):
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="lovely day")
    result = make_text_result("lovely day", "toxic", 0.99, [])
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    updated = apply_routing_decision(db_session, req, decision)

    assert updated.status == RequestStatus.AUTO_APPROVED
    assert updated.resolved_at is not None


def test_list_queue_only_returns_queued_and_uncertain(db_session, router):
    # one queued, one auto-approved (should not appear in queue)
    req1 = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="spam text")
    r1 = make_text_result("spam text", "spam", 0.80, ["spam"])
    d1 = router.route_text(r1)
    record_prediction(db_session, req1, Modality.TEXT, r1, d1.tier.value, d1.action.value)
    apply_routing_decision(db_session, req1, d1)

    req2 = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="clean text")
    r2 = make_text_result("clean text", "toxic", 0.99, [])
    d2 = router.route_text(r2)
    record_prediction(db_session, req2, Modality.TEXT, r2, d2.tier.value, d2.action.value)
    apply_routing_decision(db_session, req2, d2)

    queue = list_queue(db_session)
    assert len(queue) == 1
    assert queue[0].id == req1.id


def test_submit_review_resolves_request_and_removes_from_queue(db_session, router):
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="spam text")
    result = make_text_result("spam text", "spam", 0.80, ["spam"])
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    apply_routing_decision(db_session, req, decision)

    assert len(list_queue(db_session)) == 1

    review = submit_review(
        db_session, req.id, decision=ReviewDecision.REJECT, moderator_id="alice",
        notes="confirmed spam", corrected_labels=["spam"],
    )
    assert review.decision == ReviewDecision.REJECT
    assert len(list_queue(db_session)) == 0


def test_submit_review_on_unknown_request_raises(db_session):
    with pytest.raises(ValueError):
        submit_review(db_session, "not-a-real-id", decision=ReviewDecision.APPROVE)


def test_submit_review_twice_raises(db_session, router):
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="text")
    result = make_text_result("text", "spam", 0.80, ["spam"])
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    apply_routing_decision(db_session, req, decision)

    submit_review(db_session, req.id, decision=ReviewDecision.APPROVE)
    with pytest.raises(ValueError):
        submit_review(db_session, req.id, decision=ReviewDecision.APPROVE)


def test_export_moderator_reviewed_text_uses_corrected_labels(db_session, router):
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="win prize now")
    result = make_text_result("win prize now", "toxic", 0.55, [])  # model got it wrong / uncertain
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    apply_routing_decision(db_session, req, decision)

    submit_review(
        db_session, req.id, decision=ReviewDecision.REJECT, moderator_id="bob",
        corrected_labels=["spam"],  # moderator corrects the model's mistake
    )

    df = export_moderator_reviewed_text_for_retraining(db_session, TEXT_LABELS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["text"] == "win prize now"
    assert row["spam"] == 1
    assert row["toxic"] == 0  # corrected label overrides the (wrong) model prediction


def test_export_falls_back_to_model_prediction_without_correction(db_session, router):
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="spam text")
    result = make_text_result("spam text", "spam", 0.80, ["spam"])
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    apply_routing_decision(db_session, req, decision)

    submit_review(db_session, req.id, decision=ReviewDecision.REJECT, corrected_labels=None)

    df = export_moderator_reviewed_text_for_retraining(db_session, TEXT_LABELS)
    assert df.iloc[0]["spam"] == 1


def test_audit_log_records_full_history(db_session, router):
    from app.db.models import AuditLog

    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content="text")
    result = make_text_result("text", "spam", 0.80, ["spam"])
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    apply_routing_decision(db_session, req, decision)
    submit_review(db_session, req.id, decision=ReviewDecision.REJECT, moderator_id="alice")

    events = [
        log.event_type
        for log in db_session.query(AuditLog).filter(AuditLog.moderation_request_id == req.id).all()
    ]
    assert "request_created" in events
    assert "prediction_recorded" in events
    assert "routing_decision_applied" in events
    assert "moderator_review_submitted" in events
