"""
Shared pytest fixtures for the content moderation pipeline test suite.

Design notes:

  - Each test gets a fresh in-memory SQLite database. Rather than
    reconfiguring ``DATABASE_URL`` and re-importing the app (expensive and
    order-dependent across a test session), the ``test_db`` fixture directly
    monkeypatches ``app.db.database.engine`` / ``SessionLocal``. Because
    Python functions resolve module-level globals by name at call time
    (not at import time), every consumer of ``session_scope()`` / ``get_db()``
    — including code already imported elsewhere — transparently starts
    using the patched test database for the duration of the test.

  - Real BERT/CLIP weight downloads aren't available in CI-like sandboxes
    without network access to huggingface.co, so ``tiny_text_model`` and
    ``tiny_image_model`` build structurally-identical, randomly-initialized
    tiny models instead, exercising exactly the same code paths.
"""

from __future__ import annotations

import os

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("SECRET_KEY", "test-secret")

import numpy as np
import open_clip
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from app.core.config import settings
from app.ml.image_moderation import ImageModerationModel
from app.ml.text_moderation import TextModerationModel


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
@pytest.fixture()
def test_db(monkeypatch):
    """
    Provide a fresh in-memory SQLite database for a single test, wired in as
    the app's live database via monkeypatching.
    """
    import app.db.database as database_module
    from app.db.models import Base  # noqa: F401 ensures models are registered

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    database_module.Base.metadata.create_all(bind=test_engine)

    monkeypatch.setattr(database_module, "engine", test_engine)
    monkeypatch.setattr(database_module, "SessionLocal", TestSessionLocal)

    yield TestSessionLocal

    database_module.Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture()
def db_session(test_db):
    """A single raw Session for tests that want direct DB access."""
    session = test_db()
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Tiny ML models (no network access required)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tiny_text_model(tmp_path) -> TextModerationModel:
    """A structurally-real but randomly-initialized tiny BERT, for fast offline tests."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(
        "you people are disgusting hate free money click here now lovely day "
        "great movie spam toxic buy cheap watches win prize".split()
    )
    vocab = list(dict.fromkeys(vocab))
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab))

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    labels = settings.text_labels
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=len(labels),
        problem_type="multi_label_classification",
    )
    model = BertForSequenceClassification(config)
    return TextModerationModel(
        model=model, tokenizer=tokenizer, labels=labels, max_length=32, device="cpu",
        model_version="tiny-test",
    )


@pytest.fixture()
def tiny_image_model() -> ImageModerationModel:
    """A structurally-real but randomly-initialized tiny CLIP, for fast offline tests."""
    from app.ml.image_prompts import PromptEnsembleConfig

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompt_config = PromptEnsembleConfig(
        nsfw_prompts=["a sexually explicit image", "a pornographic image"],
        safe_prompts=["a normal photograph", "a safe image"],
        fp_mitigation_prompts=[],
    )
    return ImageModerationModel(
        model=model, preprocess=preprocess, tokenizer=tokenizer,
        labels=settings.image_labels, device="cpu", model_version="tiny-clip-test",
        prompt_config=prompt_config,
    )


@pytest.fixture()
def sample_image(tmp_path):
    """A small synthetic RGB image saved to disk, for image-pipeline tests."""
    rng = np.random.default_rng(0)
    arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
    path = tmp_path / "sample.jpg"
    Image.fromarray(arr).save(path)
    return path


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
@pytest.fixture()
def api_client(test_db, tiny_text_model, tiny_image_model):
    """A FastAPI TestClient with the DB and ML models swapped for test doubles."""
    from app.api.deps import get_image_model, get_text_model
    from app.main import app

    app.dependency_overrides[get_text_model] = lambda: tiny_text_model
    app.dependency_overrides[get_image_model] = lambda: tiny_image_model

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
