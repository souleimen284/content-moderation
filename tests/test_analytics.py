"""Tests for app.services.analytics_service (shared by GET /metrics and the dashboard, Module 8)."""

from __future__ import annotations

from app.db.models import ContentType, Modality, ReviewDecision
from app.schemas.text_moderation import LabelScore, TextModerationResult
from app.services.analytics_service import (
    get_active_model_confusion_matrices,
    get_category_distribution,
    get_confidence_scores,
    get_daily_volume,
    get_moderator_performance,
    get_overall_metrics,
)
from app.services.model_registry import register_model_version
from app.services.queue_service import (
    apply_routing_decision,
    create_moderation_request,
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


def _seed(db_session, text, top_label, score, flagged, review=None):
    router = ConfidenceRouter(auto_threshold=0.95, queue_threshold=0.70)
    req = create_moderation_request(db_session, content_type=ContentType.TEXT, text_content=text)
    result = make_text_result(text, top_label, score, flagged)
    decision = router.route_text(result)
    record_prediction(db_session, req, Modality.TEXT, result, decision.tier.value, decision.action.value)
    apply_routing_decision(db_session, req, decision)
    if review:
        submit_review(db_session, req.id, decision=review, moderator_id="alice", corrected_labels=flagged)
    return req


def test_get_overall_metrics_counts(db_session):
    _seed(db_session, "clean text", "toxic", 0.99, [])
    _seed(db_session, "spam text", "spam", 0.80, ["spam"], review=ReviewDecision.REJECT)

    stats = get_overall_metrics(db_session)
    assert stats["total_requests"] == 2
    assert stats["requests_by_status"]["auto_approved"] == 1
    assert stats["reviewed_count"] == 1
    assert stats["reviews_by_decision"]["reject"] == 1


def test_get_daily_volume_counts_todays_requests(db_session):
    _seed(db_session, "text one", "toxic", 0.99, [])
    _seed(db_session, "text two", "toxic", 0.99, [])
    volume_df = get_daily_volume(db_session, days=7)
    assert volume_df["count"].sum() == 2


def test_get_confidence_scores_returns_all_predictions(db_session):
    _seed(db_session, "a", "toxic", 0.91, [])
    _seed(db_session, "b", "spam", 0.77, ["spam"])
    scores_df = get_confidence_scores(db_session, Modality.TEXT)
    assert len(scores_df) == 2
    assert set(scores_df["top_score"].round(2)) == {0.91, 0.77}


def test_get_category_distribution_counts_flagged_labels(db_session):
    _seed(db_session, "a", "spam", 0.99, ["spam"])
    _seed(db_session, "b", "spam", 0.99, ["spam"])
    _seed(db_session, "c", "toxic", 0.99, ["toxic"])
    dist = get_category_distribution(db_session, Modality.TEXT)
    dist_map = dict(zip(dist["label"], dist["count"]))
    assert dist_map["spam"] == 2
    assert dist_map["toxic"] == 1


def test_get_category_distribution_empty_when_nothing_flagged(db_session):
    _seed(db_session, "clean", "toxic", 0.99, [])
    dist = get_category_distribution(db_session, Modality.TEXT)
    assert dist.empty


def test_get_moderator_performance_aggregates_by_moderator(db_session):
    _seed(db_session, "a", "spam", 0.80, ["spam"], review=ReviewDecision.REJECT)
    _seed(db_session, "b", "spam", 0.80, ["spam"], review=ReviewDecision.APPROVE)
    perf_df = get_moderator_performance(db_session)
    assert len(perf_df) == 1
    row = perf_df.iloc[0]
    assert row["moderator_id"] == "alice"
    assert row["total_reviews"] == 2
    assert row["reject"] == 1
    assert row["approve"] == 1


def test_get_moderator_performance_empty_when_no_reviews(db_session):
    _seed(db_session, "a", "toxic", 0.99, [])
    perf_df = get_moderator_performance(db_session)
    assert perf_df.empty


def test_get_active_model_confusion_matrices(db_session):
    metrics = {"f1_macro": 0.8, "confusion_matrix": {"toxic": {"tp": 5, "fp": 1, "tn": 10, "fn": 2}}}
    register_model_version(
        db_session, model_type=Modality.TEXT, version_tag="v1", checkpoint_path="/fake",
        metrics=metrics, training_samples=50, activate=True,
    )
    result = get_active_model_confusion_matrices(db_session)
    assert result["text"] == {"toxic": {"tp": 5, "fp": 1, "tn": 10, "fn": 2}}
    assert result["image"] is None
