"""
Text moderation model: fine-tuned BERT for multi-label content classification.

Categories (configurable via ``settings.text_labels``): toxic, spam, hate.
Each category gets an independent sigmoid output rather than a shared softmax,
since a single piece of text can belong to multiple categories simultaneously
(e.g. both "toxic" and "hate").

Label order (always consistent, index positions):
  0 = toxic
  1 = spam
  2 = hate

Usage
-----
    model = TextModerationModel.from_pretrained()
    results = model.predict(["you people are disgusting", "buy cheap watches now!!"])
    for r in results:
        print(r.top_label, r.top_score, r.predicted_labels)
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.text_moderation import LabelScore, TextModerationResult

logger = get_logger(__name__)

# Per-label cutoff for the boolean "predicted" flag. This is intentionally
# separate from the *routing* thresholds in app.core.config (auto/queue/
# uncertain), which operate on the overall top_score to decide the review
# workflow. This one just answers "is this label present at all".
DEFAULT_LABEL_DECISION_CUTOFF = 0.7


class TextModerationModel:
    """Wraps a HF sequence-classification model for multi-label text moderation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        labels: list[str],
        max_length: int = 256,
        device: str = "cpu",
        model_version: str = "unversioned",
        label_decision_cutoff: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.device = device
        self.model_version = model_version
        self.label_decision_cutoff = (
            float(label_decision_cutoff)
            if label_decision_cutoff is not None
            else float(settings.text_label_decision_cutoff)
        )

        if self.model.config.num_labels != len(self.labels):
            raise ValueError(
                f"Model head has {self.model.config.num_labels} outputs but "
                f"{len(self.labels)} labels were provided: {self.labels}"
            )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path | None = None,
        labels: list[str] | None = None,
        max_length: int | None = None,
        device: str | None = None,
    ) -> "TextModerationModel":
        """
        Load a model + tokenizer, preferring a locally fine-tuned checkpoint.

        Resolution order:
          1. Explicit ``model_name_or_path`` argument, if given.
          2. ``settings.text_model_checkpoint_dir`` if it exists on disk and
             contains a saved model (i.e. retrain.py has produced one).

        If neither exists we fall back to the base pretrained model with a
        warning, so the API can serve requests immediately. Predictions will
        improve once ``retrain.py`` produces a fine-tuned checkpoint.
        """
        labels = labels or settings.text_labels
        max_length = max_length or settings.text_model_max_length
        device = device or settings.device

        checkpoint_dir = Path(settings.text_model_checkpoint_dir)
        base_source = settings.text_model_name
        source = base_source
        version = f"{base_source}:base"
        model = None

        if model_name_or_path is not None:
            source = str(model_name_or_path)
            version = "custom"
            model = AutoModelForSequenceClassification.from_pretrained(
                source,
                num_labels=len(labels),
                problem_type="multi_label_classification",
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(source)
        elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            # Try loading the fine-tuned checkpoint; fall back to the base
            # model if the checkpoint is incompatible (e.g. different label
            # set or model architecture).
            try:
                logger.info("Loading fine-tuned text moderation checkpoint from %s", checkpoint_dir)
                tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(checkpoint_dir),
                    num_labels=len(labels),
                    problem_type="multi_label_classification",
                    low_cpu_mem_usage=True,
                )
                version = checkpoint_dir.name or "fine-tuned"
            except (RuntimeError, OSError) as exc:
                if "paging file" in str(exc).lower() or "os error 1455" in str(exc).lower():
                    raise RuntimeError(
                        f"Unable to load trained text checkpoint at {checkpoint_dir}: "
                        "Windows virtual memory is too small. Increase the paging file "
                        "or free system memory, then restart the API."
                    ) from exc
                logger.warning(
                    "Fine-tuned checkpoint at %s is incompatible (%s) — falling "
                    "back to base pretrained weights '%s'. Predictions will improve "
                    "once retrain.py produces a compatible fine-tuned checkpoint.",
                    checkpoint_dir,
                    exc,
                    base_source,
                )
                model = None  # will be loaded below

        if model is None:
            logger.warning(
                "Using base pretrained weights '%s' for text classification.",
                base_source,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_source)
            model = AutoModelForSequenceClassification.from_pretrained(
                base_source,
                num_labels=len(labels),
                problem_type="multi_label_classification",
            )

        return cls(
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            max_length=max_length,
            device=device,
            model_version=version,
            label_decision_cutoff=settings.text_label_decision_cutoff,
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict(self, texts: str | list[str]) -> list[TextModerationResult]:
        """
        Run multi-label moderation inference.

        Args:
            texts: a single string or a list of strings.

        Returns:
            One ``TextModerationResult`` per input text, each containing a
            confidence score for every configured label plus the overall
            top label/score (used downstream for confidence-based routing).
        """
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else list(texts)
        if not text_list:
            return []

        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits
        probs = torch.sigmoid(logits).cpu().numpy()

        results: list[TextModerationResult] = []
        for text, row in zip(text_list, probs):
            label_scores = [
                LabelScore(
                    label=label,
                    score=float(score),
                    predicted=bool(score >= self.label_decision_cutoff),
                )
                for label, score in zip(self.labels, row)
            ]
            top_idx = int(row.argmax())
            results.append(
                TextModerationResult(
                    text=text,
                    labels=label_scores,
                    predicted_labels=[ls.label for ls in label_scores if ls.predicted],
                    top_label=self.labels[top_idx],
                    top_score=float(row[top_idx]),
                    model_version=self.model_version,
                )
            )

        logger.info("Scored %d text item(s) with model_version=%s", len(text_list), self.model_version)
        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_pretrained(self, output_dir: str | Path) -> None:
        """Save model + tokenizer weights to ``output_dir`` (used by retrain.py)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Saved text moderation model to %s", output_dir)


_singleton: TextModerationModel | None = None


def get_text_moderation_model() -> TextModerationModel:
    """
    Process-wide singleton accessor.

    FastAPI dependency-injects this (Module 7) so the (potentially large)
    BERT model is loaded once per worker process, not once per request.
    """
    global _singleton
    if _singleton is None:
        _singleton = TextModerationModel.from_pretrained()
    return _singleton


def reload_text_moderation_model() -> TextModerationModel:
    """
    Force the singleton to reload from disk (e.g. after retrain.py promotes
    a new checkpoint). Call this instead of restarting the API process.
    """
    global _singleton
    _singleton = TextModerationModel.from_pretrained()
    logger.info("Text moderation model reloaded: version=%s", _singleton.model_version)
    return _singleton
