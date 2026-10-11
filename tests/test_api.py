"""Tests for the FastAPI endpoints (Module 7): all seven required routes."""

from __future__ import annotations


def test_health_check(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_moderate_text_success(api_client):
    response = api_client.post("/api/v1/moderate/text", json={"text": "you people are disgusting"})
    assert response.status_code == 200
    body = response.json()
    assert "request_id" in body
    assert body["prediction"]["text"] == "you people are disgusting"
    assert "routing" in body
    assert body["routing"]["source"] == "text"


def test_moderate_text_rejects_empty_string(api_client):
    response = api_client.post("/api/v1/moderate/text", json={"text": "   "})
    assert response.status_code == 422


def test_moderate_image_success(api_client, sample_image):
    with open(sample_image, "rb") as f:
        response = api_client.post(
            "/api/v1/moderate/image", files={"image": ("sample.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    body = response.json()
    assert "request_id" in body
    assert body["routing"]["source"] == "image"


def test_moderate_content_with_both_text_and_image(api_client, sample_image):
    with open(sample_image, "rb") as f:
        response = api_client.post(
            "/api/v1/moderate/content",
            data={"text": "free money click here now"},
            files={"image": ("sample.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["source"] == "combined"
    assert body["text_prediction"] is not None
    assert body["image_prediction"] is not None


def test_moderate_content_text_only(api_client):
    response = api_client.post("/api/v1/moderate/content", data={"text": "what a lovely day"})
    assert response.status_code == 200
    body = response.json()
    assert body["image_prediction"] is None
    assert body["text_prediction"] is not None


def test_moderate_content_requires_something(api_client):
    response = api_client.post("/api/v1/moderate/content", data={})
    assert response.status_code == 422


def test_moderation_queue_lists_items(api_client):
    # A mid-confidence text prediction should land in the queue.
    api_client.post("/api/v1/moderate/text", json={"text": "win a free prize now click here"})
    response = api_client.get("/api/v1/moderation_queue")
    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert body["limit"] == 50


def test_moderation_queue_invalid_status_filter(api_client):
    response = api_client.get("/api/v1/moderation_queue", params={"status": "not_a_real_status"})
    assert response.status_code == 422


def test_review_flow_end_to_end(api_client):
    submit_resp = api_client.post("/api/v1/moderate/text", json={"text": "buy cheap watches now"})
    request_id = submit_resp.json()["request_id"]

    queue_resp = api_client.get("/api/v1/moderation_queue")
    queue_ids = [item["id"] for item in queue_resp.json()["items"]]

    if request_id in queue_ids:
        review_resp = api_client.post(
            "/api/v1/review",
            json={
                "moderation_request_id": request_id,
                "decision": "reject",
                "moderator_id": "test_moderator",
                "notes": "confirmed",
                "corrected_labels": ["spam"],
            },
        )
        assert review_resp.status_code == 200
        assert review_resp.json()["decision"] == "reject"

        # Reviewing again should 404.
        second_resp = api_client.post(
            "/api/v1/review", json={"moderation_request_id": request_id, "decision": "approve"}
        )
        assert second_resp.status_code == 404


def test_review_invalid_decision_value(api_client):
    response = api_client.post(
        "/api/v1/review", json={"moderation_request_id": "some-id", "decision": "maybe"}
    )
    assert response.status_code == 422


def test_review_unknown_request_returns_404(api_client):
    response = api_client.post(
        "/api/v1/review", json={"moderation_request_id": "does-not-exist", "decision": "approve"}
    )
    assert response.status_code == 404


def test_metrics_endpoint(api_client):
    api_client.post("/api/v1/moderate/text", json={"text": "hello world"})
    response = api_client.get("/api/v1/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] >= 1
    assert "requests_by_status" in body
    assert "reviews_by_decision" in body


def test_unprefixed_health_and_metrics_also_work(api_client):
    assert api_client.get("/health").status_code == 200
    assert api_client.get("/metrics").status_code == 200


def test_root_endpoint(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()
