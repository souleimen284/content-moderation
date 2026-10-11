"""Tests for app.ml.text_moderation (Module 2)."""

from __future__ import annotations

from app.core.config import settings


def test_predict_returns_one_result_per_input(tiny_text_model):
    results = tiny_text_model.predict(["hello there", "another message"])
    assert len(results) == 2
    for r in results:
        assert set(r.labels[i].label for i in range(len(r.labels))) == set(settings.text_labels)


def test_predict_single_string_returns_list_of_one(tiny_text_model):
    results = tiny_text_model.predict("just one text")
    assert isinstance(results, list)
    assert len(results) == 1


def test_predict_empty_list_returns_empty(tiny_text_model):
    assert tiny_text_model.predict([]) == []


def test_each_label_score_in_valid_range(tiny_text_model):
    results = tiny_text_model.predict("some text to score")
    for label_score in results[0].labels:
        assert 0.0 <= label_score.score <= 1.0


def test_predicted_labels_match_cutoff(tiny_text_model):
    result = tiny_text_model.predict("some text")[0]
    predicted_from_labels = {ls.label for ls in result.labels if ls.predicted}
    assert predicted_from_labels == set(result.predicted_labels)
    for ls in result.labels:
        assert ls.predicted == (ls.score >= tiny_text_model.label_decision_cutoff)


def test_top_label_is_argmax(tiny_text_model):
    result = tiny_text_model.predict("some text")[0]
    best = max(result.labels, key=lambda ls: ls.score)
    assert result.top_label == best.label
    assert result.top_score == best.score


def test_model_version_propagates(tiny_text_model):
    result = tiny_text_model.predict("text")[0]
    assert result.model_version == tiny_text_model.model_version


def test_default_label_cutoff_is_conservative():
    assert hasattr(settings, "text_label_decision_cutoff")
    assert settings.text_label_decision_cutoff >= 0.6
    assert settings.text_label_decision_cutoff <= 0.8


def test_save_and_reload_pretrained(tmp_path, tiny_text_model):
    from app.ml.text_moderation import TextModerationModel

    tiny_text_model.save_pretrained(tmp_path)
    reloaded = TextModerationModel.from_pretrained(
        model_name_or_path=tmp_path, labels=settings.text_labels, max_length=32, device="cpu"
    )
    assert reloaded.labels == tiny_text_model.labels
    result = reloaded.predict("hello there")[0]
    assert result.top_label in settings.text_labels


def test_mismatched_label_count_raises():
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_file = Path(tmpdir) / "vocab.txt"
        vocab_file.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
        config = BertConfig(
            vocab_size=tokenizer.vocab_size, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=1, intermediate_size=32, num_labels=5,
            problem_type="multi_label_classification",
        )
        model = BertForSequenceClassification(config)

        from app.ml.text_moderation import TextModerationModel
        import pytest

        with pytest.raises(ValueError):
            TextModerationModel(model=model, tokenizer=tokenizer, labels=settings.text_labels)
