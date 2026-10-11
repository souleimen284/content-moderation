from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from app.ml.image_moderation import ImageModerationModel


class FakeNudeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        assert isinstance(image, np.ndarray)
        return self.detections


def build_model(detections):
    return ImageModerationModel(
        model=torch.nn.Identity(),
        preprocess=lambda image: image,
        tokenizer=lambda prompts: prompts,
        labels=[],
        model_version="NudeNet-320n-test",
        nudenet_detector=FakeNudeDetector(detections),
        safe_threshold=0.35,
        nsfw_threshold=0.75,
    )


def test_nudenet_ignores_covered_body_parts():
    model = build_model([{"class": "BELLY_COVERED", "score": 0.99}])
    result = model.predict(Image.new("RGB", (8, 8)))[0]
    assert result.decision == "SAFE"
    assert result.nsfw_score == 0.0
    assert result.predicted_labels == []
    assert result.categories == {"BELLY_COVERED": 0.99}


def test_nudenet_exposed_genitalia_is_nsfw():
    model = build_model([{"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.91}])
    result = model.predict(Image.new("RGB", (8, 8)))[0]
    assert result.decision == "NSFW"
    assert result.nsfw_score == 0.91
    assert result.predicted_labels == ["FEMALE_GENITALIA_EXPOSED"]