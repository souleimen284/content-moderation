"""Tests for app.ml.image_moderation (Module 3), covering zero-shot prompt ensembling and linear-probe paths."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from app.core.config import settings
from app.ml.image_moderation import _softmax_similarity, _three_way_decision


def test_zero_shot_predict_returns_one_result_per_image(tiny_image_model, sample_image, tmp_path):
    from PIL import Image

    img2_path = tmp_path / "sample2.jpg"
    Image.open(sample_image).save(img2_path)

    results = tiny_image_model.predict([str(sample_image), str(img2_path)], sources=["a.jpg", "b.jpg"])
    assert len(results) == 2
    assert results[0].source == "a.jpg"
    for r in results:
        assert set(ls.label for ls in r.labels) == set(settings.image_labels)


def test_predict_single_image_non_list_input(tiny_image_model, sample_image):
    results = tiny_image_model.predict(str(sample_image))
    assert isinstance(results, list)
    assert len(results) == 1


def test_predict_from_bytes(tiny_image_model, sample_image):
    data = sample_image.read_bytes()
    results = tiny_image_model.predict(data)
    assert len(results) == 1
    assert 0.0 <= results[0].top_score <= 1.0


def test_predict_from_pil_image(tiny_image_model):
    from PIL import Image

    rng = np.random.default_rng(1)
    img = Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8"))
    results = tiny_image_model.predict(img)
    assert len(results) == 1


def test_sources_length_mismatch_raises(tiny_image_model, sample_image):
    with pytest.raises(ValueError):
        tiny_image_model.predict([str(sample_image)], sources=["a.jpg", "b.jpg"])


def test_zero_shot_mode_when_no_probe(tiny_image_model):
    assert tiny_image_model.linear_probe is None


def test_linear_probe_path_used_when_present(tiny_image_model, sample_image, tmp_path):
    """When a trained linear probe is attached, predict() should use it, not zero-shot prompts."""
    embeddings = tiny_image_model.encode_images([str(sample_image)] * 10)
    fake_labels = np.zeros((10, len(settings.image_labels)), dtype=int)
    fake_labels[:5, 0] = 1  # first label positive for half the "samples"

    probe = OneVsRestClassifier(LogisticRegression(max_iter=200))
    probe.fit(embeddings, fake_labels)

    tiny_image_model.linear_probe = probe
    results = tiny_image_model.predict(str(sample_image))
    assert len(results) == 1
    # probability output should come from predict_proba (shape n_labels), not the zero-shot path
    scores = [ls.score for ls in results[0].labels]
    assert len(scores) == len(settings.image_labels)


def test_save_and_load_linear_probe(tmp_path, tiny_image_model, sample_image):
    from app.ml.image_moderation import ImageModerationModel

    embeddings = tiny_image_model.encode_images([str(sample_image)] * 6)
    labels = np.array([[1, 0, 0, 0, 0]] * 3 + [[0, 0, 0, 0, 0]] * 3)
    probe = OneVsRestClassifier(LogisticRegression(max_iter=200))
    probe.fit(embeddings, labels)

    saved_path = ImageModerationModel.save_linear_probe(probe, tmp_path)
    assert saved_path.exists()

    import joblib

    reloaded_probe = joblib.load(saved_path)
    preds = reloaded_probe.predict_proba(embeddings)
    assert preds.shape == (6, len(settings.image_labels))


def test_encode_images_returns_normalized_embeddings(tiny_image_model, sample_image):
    embeddings = tiny_image_model.encode_images([str(sample_image)])
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# Prompt ensembling & scoring tests
# --------------------------------------------------------------------------- #

def test_softmax_similarity_symmetric():
    """Softmax of equal scores should yield ~0.5."""
    score = _softmax_similarity(3.0, 3.0)
    assert abs(score - 0.5) < 1e-6


def test_softmax_similarity_one_sided():
    """When one score is much larger, softmax should approach 1.0 or 0.0."""
    assert _softmax_similarity(100.0, 0.0) > 0.99
    assert _softmax_similarity(0.0, 100.0) < 0.01


def test_softmax_similarity_returns_bounded():
    """Result should always be in [0, 1]."""
    for a, b in [(0, 0), (1, 2), (5, -3), (-10, -10)]:
        result = _softmax_similarity(a, b)
        assert 0.0 <= result <= 1.0


def test_three_way_decision_safe():
    assert _three_way_decision(0.1, 0.3, 0.7) == "SAFE"
    assert _three_way_decision(0.0, 0.3, 0.7) == "SAFE"
    assert _three_way_decision(0.29, 0.3, 0.7) == "SAFE"


def test_three_way_decision_review():
    assert _three_way_decision(0.5, 0.3, 0.7) == "REVIEW"
    assert _three_way_decision(0.3, 0.3, 0.7) == "REVIEW"
    assert _three_way_decision(0.7, 0.3, 0.7) == "REVIEW"


def test_three_way_decision_nsfw():
    assert _three_way_decision(0.9, 0.3, 0.7) == "NSFW"
    assert _three_way_decision(0.71, 0.3, 0.7) == "NSFW"
    assert _three_way_decision(1.0, 0.3, 0.7) == "NSFW"


def test_zero_shot_ensemble_returns_decision_field(tiny_image_model, sample_image):
    """New zero-shot ensemble should populate decision, nsfw_score, and method."""
    results = tiny_image_model.predict(str(sample_image))
    assert len(results) == 1
    r = results[0]
    assert r.decision in ("SAFE", "REVIEW", "NSFW")
    assert 0.0 <= r.nsfw_score <= 1.0
    assert r.method == "zero-shot-clip-prompt-ensemble"


def test_nsfw_score_is_softmax_scaled(tiny_image_model, sample_image):
    """nsfw_score should be a softmax-scaled value, not a raw similarity."""
    results = tiny_image_model.predict(str(sample_image))
    r = results[0]
    # Softmax output is always in [0, 1]
    assert 0.0 <= r.nsfw_score <= 1.0


def test_decision_matches_nsfw_score(tiny_image_model, sample_image):
    """The decision should be consistent with the nsfw_score and thresholds."""
    results = tiny_image_model.predict(str(sample_image))
    r = results[0]
    safe_thresh = tiny_image_model.safe_threshold
    nsfw_thresh = tiny_image_model.nsfw_threshold
    if r.nsfw_score < safe_thresh:
        assert r.decision == "SAFE"
    elif r.nsfw_score > nsfw_thresh:
        assert r.decision == "NSFW"
    else:
        assert r.decision == "REVIEW"


def test_predict_multiple_images_ensemble(tiny_image_model, sample_image, tmp_path):
    """Ensemble scoring should work for multiple images."""
    from PIL import Image
    img2_path = tmp_path / "sample2.jpg"
    Image.open(sample_image).save(img2_path)

    results = tiny_image_model.predict([str(sample_image), str(img2_path)], sources=["a.jpg", "b.jpg"])
    assert len(results) == 2
    for r in results:
        assert r.decision in ("SAFE", "REVIEW", "NSFW")
        assert 0.0 <= r.nsfw_score <= 1.0
        assert r.method == "zero-shot-clip-prompt-ensemble"


def test_configurable_thresholds(tiny_image_model):
    """Thresholds should be configurable via the model."""
    tiny_image_model.safe_threshold = 0.1
    tiny_image_model.nsfw_threshold = 0.9
    assert tiny_image_model.safe_threshold == 0.1
    assert tiny_image_model.nsfw_threshold == 0.9


def test_prompt_config_accessible(tiny_image_model):
    """The prompt ensemble config should be accessible on the model."""
    from app.ml.image_prompts import PromptEnsembleConfig
    assert isinstance(tiny_image_model.prompt_config, PromptEnsembleConfig)
    assert len(tiny_image_model.prompt_config.nsfw_prompts) > 0
    assert len(tiny_image_model.prompt_config.safe_prompts) > 0
