"""Tests for app.services.model_registry and retrain.py (Module 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.db.models import Modality
from app.services import model_registry


class TestModelRegistry:
    def test_register_first_version_unconditionally_promoted(self, db_session):
        active, promoted = model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="v1",
            candidate_checkpoint_path="/fake/v1", candidate_metrics={"f1_macro": 0.5},
            training_samples=10,
        )
        assert promoted is True
        assert active.version_tag == "v1"
        assert active.is_active is True

    def test_worse_candidate_is_not_promoted(self, db_session):
        model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="v1",
            candidate_checkpoint_path="/fake/v1", candidate_metrics={"f1_macro": 0.8},
            training_samples=10,
        )
        active, promoted = model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="v2_worse",
            candidate_checkpoint_path="/fake/v2", candidate_metrics={"f1_macro": 0.5},
            training_samples=10,
        )
        assert promoted is False
        assert active.version_tag == "v1"

    def test_better_candidate_is_promoted(self, db_session):
        model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="v1",
            candidate_checkpoint_path="/fake/v1", candidate_metrics={"f1_macro": 0.5},
            training_samples=10,
        )
        active, promoted = model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="v2_better",
            candidate_checkpoint_path="/fake/v2", candidate_metrics={"f1_macro": 0.9},
            training_samples=10,
        )
        assert promoted is True
        assert active.version_tag == "v2_better"

    def test_exactly_one_active_version_per_modality(self, db_session):
        from app.db.models import ModelVersion

        model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="v1",
            candidate_checkpoint_path="/fake/v1", candidate_metrics={"f1_macro": 0.5},
            training_samples=10,
        )
        model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="v2",
            candidate_checkpoint_path="/fake/v2", candidate_metrics={"f1_macro": 0.9},
            training_samples=10,
        )
        active_versions = (
            db_session.query(ModelVersion)
            .filter(ModelVersion.model_type == Modality.TEXT, ModelVersion.is_active.is_(True))
            .all()
        )
        assert len(active_versions) == 1
        assert active_versions[0].version_tag == "v2"

    def test_modalities_are_independent(self, db_session):
        model_registry.promote_if_better(
            db_session, model_type=Modality.TEXT, candidate_version_tag="text_v1",
            candidate_checkpoint_path="/fake/t1", candidate_metrics={"f1_macro": 0.9},
            training_samples=10,
        )
        active_image, promoted_image = model_registry.promote_if_better(
            db_session, model_type=Modality.IMAGE, candidate_version_tag="image_v1",
            candidate_checkpoint_path="/fake/i1", candidate_metrics={"f1_macro": 0.1},
            training_samples=10,
        )
        # No prior active IMAGE model, so this promotes unconditionally
        # regardless of the unrelated TEXT model's score.
        assert promoted_image is True
        assert active_image.version_tag == "image_v1"

    def test_get_active_model_version_returns_none_when_absent(self, db_session):
        assert model_registry.get_active_model_version(db_session, Modality.TEXT) is None


class TestTextDataMerging:
    def test_merge_deduplicates_on_text_with_moderator_correction_winning(self, db_session, monkeypatch, tmp_path):
        import retrain

        monkeypatch.setattr(retrain.settings, "retrain_data_dir", tmp_path)
        monkeypatch.setattr(retrain.settings, "text_labels", ["toxic", "spam"])

        base_csv = tmp_path / "base.csv"
        base_df = pd.DataFrame(
            {"text": ["hello world", "click here now"], "toxic": [0, 0], "spam": [0, 1]}
        )
        base_df.to_csv(base_csv, index=False)

        # Simulate a moderator having reviewed "click here now" and corrected it to toxic=1.
        def fake_export(db, labels):
            return pd.DataFrame({"text": ["click here now"], "toxic": [1], "spam": [0]})

        monkeypatch.setattr(retrain, "export_moderator_reviewed_text_for_retraining", fake_export)

        merged_path, new_row_count = retrain.merge_text_training_data(base_csv)
        merged_df = pd.read_csv(merged_path)

        assert new_row_count == 1
        assert len(merged_df) == 2  # deduplicated, not appended as a 3rd row
        corrected_row = merged_df[merged_df["text"] == "click here now"].iloc[0]
        assert corrected_row["toxic"] == 1  # moderator correction won


class TestRetrainOrchestration:
    def test_run_promotes_first_text_model_unconditionally(self, db_session, monkeypatch, tmp_path):
        import retrain

        monkeypatch.setattr(retrain.settings, "retrain_data_dir", tmp_path)
        monkeypatch.setattr(retrain.settings, "text_model_checkpoint_dir", tmp_path / "live_text")

        def fake_export(db, labels):
            return pd.DataFrame(columns=["text", *labels])

        monkeypatch.setattr(retrain, "export_moderator_reviewed_text_for_retraining", fake_export)

        def fake_train_text(cfg):
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            (cfg.output_dir / "fake_weights.bin").write_text("fake")
            return {"f1_macro": 0.7}

        monkeypatch.setattr(retrain, "train_text", fake_train_text)
        monkeypatch.setattr(retrain, "reload_text_moderation_model", lambda: None)

        result = retrain.run_text_retraining(
            base_data_path=Path("data/sample_text_moderation.csv"), min_new_samples=100
        )
        assert result.ran is True
        assert result.promoted is True
        assert result.metrics["f1_macro"] == 0.7

    def test_run_skips_image_when_no_reviewed_data(self, db_session, monkeypatch, tmp_path):
        import retrain

        # No moderator-reviewed image data exists in this fresh test DB, so
        # the image trainer should return None (a legitimate "not enough
        # data yet" outcome, not an error).
        result = retrain.run_image_retraining(min_samples=10)
        assert result.ran is False
        assert "insufficient" in result.reason

    def test_run_handles_text_exception_gracefully(self, db_session, monkeypatch, tmp_path):
        import retrain

        def failing_train_text(cfg):
            raise RuntimeError("simulated training failure")

        monkeypatch.setattr(retrain, "merge_text_training_data", lambda path: (tmp_path / "x.csv", 0))
        monkeypatch.setattr(
            retrain, "run_text_retraining",
            lambda base_data_path, min_new_samples: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        results = retrain.run(
            text_data_path=Path("data/sample_text_moderation.csv"), skip_text=False, skip_image=True
        )
        text_result = next(r for r in results if r.modality == "text")
        assert text_result.ran is False
        assert "exception" in text_result.reason
