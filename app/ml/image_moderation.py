"""
Image moderation model: CLIP-based zero-shot image classification with prompt ensembling.

This module implements an enhanced zero-shot image moderation system that uses:

  1. A strong pretrained OpenCLIP model (default: ViT-L/14)
  2. Prompt ensembling — multiple descriptive prompts per category
  3. Proper embedding normalization
  4. Softmax-based scoring (pseudo-probability, NOT a true calibrated probability)
  5. Three-way SAFE / REVIEW / NSFW routing with configurable thresholds
  6. False-positive mitigation prompts for beaches, swimsuits, fitness, etc.

Architecture:
  Application startup:
    CLIP model → tokenize ALL prompts → generate text embeddings → normalize → CACHE

  Per image:
    image → preprocess → CLIP image encoder → normalize → compare with cached
    prompt embeddings → aggregate similarities → softmax → SAFE / REVIEW / NSFW

Usage:
    model = ImageModerationModel.from_pretrained()
    results = model.predict(["path/to/image.jpg"])
    for r in results:
        print(r.decision, r.nsfw_score, r.model_version)
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import open_clip
import torch
import yaml
from nudenet import NudeDetector
from PIL import Image

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml.image_prompts import PromptEnsembleConfig, load_prompt_config
from app.schemas.image_moderation import ImageLabelScore, ImageModerationResult

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Defaults (overridable via config / env vars)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_NAME = "ViT-L-14"
DEFAULT_MODEL_PRETRAINED = "openai"

# Pseudo-probability thresholds for three-way routing.
# NOTE: These are NOT calibrated probabilities. They should eventually be
# tuned using a validation dataset of real images.
DEFAULT_IMAGE_SAFE_THRESHOLD = 0.30   # score < this → SAFE
DEFAULT_IMAGE_NSFW_THRESHOLD = 0.70   # score > this → NSFW
# Between the two → REVIEW (uncertain)

# Pairwise CLIP scores are not calibrated probabilities; use a conservative
# cutoff for per-label flagging.
DEFAULT_LABEL_DECISION_CUTOFF = 0.8
NUDENET_RISK_WEIGHTS = {
    "FEMALE_GENITALIA_EXPOSED": 1.0,
    "MALE_GENITALIA_EXPOSED": 1.0,
    "ANUS_EXPOSED": 1.0,
    "FEMALE_BREAST_EXPOSED": 0.85,
    "BUTTOCKS_EXPOSED": 0.75,
}
SECONDARY_CLIP_LABELS = ["violence", "weapons", "hate_symbols", "graphic_content"]

ImageInput = str | Path | bytes | Image.Image


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_yaml_config() -> dict:
    """Load the image_model section from configs/model_config.yaml."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("image_model", {})
    except FileNotFoundError:
        logger.warning("model_config.yaml not found at %s; using defaults.", config_path)
        return {}


def _to_pil_image(image: ImageInput) -> Image.Image:
    """Normalize any supported input type into an RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)}")


def _softmax_similarity(score_a: float, score_b: float) -> float:
    """
    Convert two similarity scores into a pseudo-probability via softmax.

    This is NOT a true calibrated probability. It produces a value in [0, 1]
    that represents the relative confidence that the image matches class A
    vs class B. The actual numerical value depends on the scale of the
    similarity scores and should not be interpreted as a literal probability.

    Returns: probability-like score for class A (i.e. 1 - score for class B).
    """
    # Numerically stable softmax over two values
    max_s = max(score_a, score_b)
    exp_a = math.exp(score_a - max_s)
    exp_b = math.exp(score_b - max_s)
    return exp_a / (exp_a + exp_b)


def _three_way_decision(
    score: float,
    safe_threshold: float,
    nsfw_threshold: float,
) -> Literal["SAFE", "REVIEW", "NSFW"]:
    """
    Map a pseudo-probability NSFW score to a three-way moderation decision.

    Args:
        score: NSFW pseudo-probability in [0, 1] from prompt ensemble softmax.
        safe_threshold: score below this → SAFE.
        nsfw_threshold: score above this → NSFW.
        Between the two → REVIEW.

    NOTE: These thresholds should be calibrated using a validation dataset.
    The defaults are conservative starting points.
    """
    if score < safe_threshold:
        return "SAFE"
    if score > nsfw_threshold:
        return "NSFW"
    return "REVIEW"


# --------------------------------------------------------------------------- #
# Main model class
# --------------------------------------------------------------------------- #

class ImageModerationModel:
    """Wraps an OpenCLIP model for zero-shot image moderation with prompt ensembling."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess,
        tokenizer,
        labels: list[str],
        device: str = "cpu",
        model_version: str = "unversioned",
        label_decision_cutoff: float = DEFAULT_LABEL_DECISION_CUTOFF,
        prompt_config: PromptEnsembleConfig | None = None,
        safe_threshold: float = DEFAULT_IMAGE_SAFE_THRESHOLD,
        nsfw_threshold: float = DEFAULT_IMAGE_NSFW_THRESHOLD,
        linear_probe=None,
        nudenet_detector: NudeDetector | None = None,
        secondary_clip: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.labels = labels
        self.device = device
        self.model_version = model_version
        self.label_decision_cutoff = label_decision_cutoff
        self.prompt_config = prompt_config or PromptEnsembleConfig()
        self.safe_threshold = safe_threshold
        self.nsfw_threshold = nsfw_threshold
        self.linear_probe = linear_probe
        self.nudenet_detector = nudenet_detector
        self.secondary_clip = secondary_clip

        #: Cached text embeddings — computed once at startup, reused per request.
        self._category_text_features: torch.Tensor | None = None
        self._nsfw_text_features: torch.Tensor | None = None
        self._safe_text_features: torch.Tensor | None = None

        if linear_probe is None and (nudenet_detector is None or secondary_clip):
            self._precompute_text_embeddings()

        logger.info(
            "ImageModerationModel initialized: model=%s, device=%s, "
            "nsfw_prompts=%d, safe_prompts=%d, fp_mitigation=%d, "
            "safe_threshold=%.2f, nsfw_threshold=%.2f",
            model_version,
            device,
            len(self.prompt_config.nsfw_prompts),
            len(self.prompt_config.safe_prompts),
            len(self.prompt_config.fp_mitigation_prompts),
            safe_threshold,
            nsfw_threshold,
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        labels: list[str] | None = None,
        device: str | None = None,
    ) -> "ImageModerationModel":
        """
        Load OpenCLIP weights, preferring a locally fine-tuned checkpoint.

        Resolution order:
          1. A fine-tuned state_dict under settings.image_model_checkpoint_dir
          2. The base pretrained OpenCLIP weights
        """
        yaml_cfg = _load_yaml_config()

        # Model name / pretrained can come from env vars OR yaml config
        model_name = settings.image_model_name or yaml_cfg.get("architecture", DEFAULT_MODEL_NAME)
        model_pretrained = settings.image_model_pretrained or yaml_cfg.get("pretrained", DEFAULT_MODEL_PRETRAINED)
        labels = labels or settings.image_labels
        clip_labels = SECONDARY_CLIP_LABELS if settings.image_model_provider == "ensemble" else labels
        device = device or settings.device

        # Load prompt configuration from yaml
        prompt_config = load_prompt_config(yaml_cfg)

        # Load thresholds from settings, fall back to yaml, then defaults
        safe_threshold = float(yaml_cfg.get("safe_threshold", DEFAULT_IMAGE_SAFE_THRESHOLD))
        nsfw_threshold = float(yaml_cfg.get("nsfw_threshold", DEFAULT_IMAGE_NSFW_THRESHOLD))
        label_cutoff = float(yaml_cfg.get("label_decision_cutoff", DEFAULT_LABEL_DECISION_CUTOFF))

        # Allow env overrides for thresholds
        safe_threshold = float(settings.image_safe_threshold) if hasattr(settings, 'image_safe_threshold') and settings.image_safe_threshold is not None else safe_threshold
        nsfw_threshold = float(settings.image_nsfw_threshold) if hasattr(settings, 'image_nsfw_threshold') and settings.image_nsfw_threshold is not None else nsfw_threshold

        if settings.image_model_provider == "nudenet":
            return cls(
                model=torch.nn.Identity(),
                preprocess=lambda image: image,
                tokenizer=lambda prompts: prompts,
                labels=[],
                device=device,
                model_version="NudeNet-320n",
                label_decision_cutoff=label_cutoff,
                safe_threshold=safe_threshold,
                nsfw_threshold=nsfw_threshold,
                nudenet_detector=NudeDetector(),
            )

        logger.info("Loading OpenCLIP model: %s (pretrained=%s)", model_name, model_pretrained)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=model_pretrained,
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        version = f"{model_name}:{model_pretrained}"

        # Check for fine-tuned checkpoint
        checkpoint_dir = Path(settings.image_model_checkpoint_dir)
        checkpoint_file = checkpoint_dir / "clip_finetuned.pt"
        if checkpoint_file.exists():
            logger.info("Loading fine-tuned CLIP checkpoint from %s", checkpoint_file)
            state_dict = torch.load(checkpoint_file, map_location="cpu")
            model.load_state_dict(state_dict)
            version = checkpoint_dir.name or "fine-tuned"
        else:
            logger.info(
                "Using base pretrained weights '%s' (%s) for zero-shot classification.",
                model_name,
                model_pretrained,
            )

        # Check for linear probe
        linear_probe = None
        probe_file = checkpoint_dir / "linear_probe.joblib"
        if probe_file.exists():
            logger.info("Loading trained linear-probe classifier from %s", probe_file)
            linear_probe = joblib.load(probe_file)
            version = f"{version}+linear_probe"

        return cls(
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            labels=clip_labels,
            device=device,
            model_version=version,
            label_decision_cutoff=label_cutoff,
            prompt_config=prompt_config,
            safe_threshold=safe_threshold,
            nsfw_threshold=nsfw_threshold,
            linear_probe=linear_probe,
            nudenet_detector=NudeDetector() if settings.image_model_provider == "ensemble" else None,
            secondary_clip=settings.image_model_provider == "ensemble",
        )

    # ------------------------------------------------------------------ #
    # Pre-compute text embeddings (called once at startup)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _precompute_text_embeddings(self) -> None:
        """
        Pre-compute and cache ALL text embeddings at startup.

        This avoids recomputing text embeddings for every request.
        Only the image embedding needs to be computed per-request.
        """
        # 1. Category text features (for per-label scoring)
        self._category_text_features = self._encode_category_prompts()

        # 2. NSFW ensemble text features (for binary SAFE/REVIEW/NSFW)
        nsfw_prompts = self.prompt_config.get_nsfw_prompts()
        safe_prompts = self.prompt_config.get_all_safe_prompts()

        self._nsfw_text_features = self._encode_prompts(nsfw_prompts)
        self._safe_text_features = self._encode_prompts(safe_prompts)

        logger.info(
            "Pre-computed text embeddings: %d category prompt pairs, "
            "%d NSFW prompts, %d SAFE prompts (incl. %d FP mitigation)",
            len(self.labels),
            len(nsfw_prompts),
            len(safe_prompts),
            len(self.prompt_config.fp_mitigation_prompts),
        )

    @torch.inference_mode()
    def _encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Tokenize and encode a list of prompts into normalized text embeddings."""
        tokens = self.tokenizer(prompts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.inference_mode()
    def _encode_category_prompts(self) -> torch.Tensor:
        """
        Pre-compute normalized text embeddings for every label's positive and
        negative prompts. Shape: (num_labels, 2, embed_dim).
        """
        all_features = []
        for label in self.labels:
            cat_prompts = self.prompt_config.category_prompts.get(label)
            if cat_prompts:
                pos_prompts = cat_prompts.get("positive", [f"a photo containing {label.replace('_', ' ')}"])
                neg_prompts = cat_prompts.get("negative", [f"a safe photo with no {label.replace('_', ' ')}"])
            else:
                readable = label.replace("_", " ")
                pos_prompts = [f"a photo containing {readable}"]
                neg_prompts = [f"a safe photo with no {readable}"]

            # Average over multiple prompts per class
            pos_features = self._encode_prompts(pos_prompts)  # (N_pos, D)
            neg_features = self._encode_prompts(neg_prompts)  # (N_neg, D)

            pos_avg = pos_features.mean(dim=0, keepdim=True)  # (1, D)
            neg_avg = neg_features.mean(dim=0, keepdim=True)  # (1, D)

            # Re-normalize the averaged vectors
            pos_avg = pos_avg / pos_avg.norm(dim=-1, keepdim=True)
            neg_avg = neg_avg / neg_avg.norm(dim=-1, keepdim=True)

            all_features.append(torch.cat([pos_avg, neg_avg], dim=0))  # (2, D)

        return torch.stack(all_features)  # (num_labels, 2, D)

    # ------------------------------------------------------------------ #
    # Image encoding
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def encode_images(self, images: list[ImageInput]) -> np.ndarray:
        """
        Encode a batch of images into normalized CLIP embeddings.

        Exposed separately from predict() so scripts/train_image_model.py
        can reuse the exact same frozen feature extractor.
        """
        pil_images = [_to_pil_image(img) for img in images]
        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        image_features = self.model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict(
        self,
        images: ImageInput | list[ImageInput],
        sources: list[str] | None = None,
    ) -> list[ImageModerationResult]:
        """
        Run moderation inference on one or more images.

        If a trained linear probe is loaded, scores come from that classifier.
        Otherwise, uses zero-shot prompt ensembling for robust classification.

        Args:
            images: a single image (path/bytes/PIL.Image) or a list of them.
            sources: optional display names/identifiers, one per image.

        Returns:
            One ImageModerationResult per input image with per-label scores,
            NSFW pseudo-probability, three-way decision, and metadata.
        """
        single_input = not isinstance(images, list)
        image_list = [images] if single_input else images
        if not image_list:
            return []

        if sources is None:
            sources = [f"image_{i}" for i in range(len(image_list))]
        elif len(sources) != len(image_list):
            raise ValueError("sources must have the same length as images")

        if self.linear_probe is not None:
            results = self._predict_with_linear_probe(image_list, sources)
        elif self.nudenet_detector is not None:
            results = self._predict_nudenet(image_list, sources)
        else:
            results = self._predict_zero_shot_ensemble(image_list, sources)

        logger.info(
            "Scored %d image(s) with model_version=%s (mode=%s)",
            len(image_list),
            self.model_version,
            (
                "linear-probe"
                if self.linear_probe is not None
                else "nudenet+clip"
                if self.nudenet_detector is not None and self.secondary_clip
                else "nudenet"
                if self.nudenet_detector is not None
                else "zero-shot-ensemble"
            ),
        )
        return results

    def _predict_nudenet(
        self, image_list: list[ImageInput], sources: list[str]
    ) -> list[ImageModerationResult]:
        """Run NudeNet detection and derive a conservative NSFW score."""
        results: list[ImageModerationResult] = []
        for image, source in zip(image_list, sources):
            pil_image = _to_pil_image(image)
            detector_input = image if isinstance(image, (str, Path)) else np.asarray(pil_image)
            detections = self.nudenet_detector.detect(detector_input)
            categories: dict[str, float] = {}
            for detection in detections:
                label = detection["class"]
                categories[label] = max(categories.get(label, 0.0), float(detection["score"]))

            risk_scores = {
                label: score * NUDENET_RISK_WEIGHTS[label]
                for label, score in categories.items()
                if label in NUDENET_RISK_WEIGHTS
            }
            secondary_predicted: set[str] = set()
            if self.secondary_clip:
                clip_result = self._predict_zero_shot_ensemble([image], [source])[0]
                clip_scores = {item.label: item.score for item in clip_result.labels}
                categories.update(clip_scores)
                secondary_predicted = {
                    label for label, score in clip_scores.items()
                    if score >= self.label_decision_cutoff
                }
                # CLIP is deliberately only a secondary signal here. Its
                # pseudo-probability is blended toward neutral before routing.
                clip_weight = float(settings.image_clip_weight)
                for label, score in clip_scores.items():
                    if score >= self.label_decision_cutoff:
                        risk_scores[label] = 0.5 + clip_weight * (score - 0.5)
            nsfw_score = min(1.0, max(risk_scores.values(), default=0.0))
            decision = _three_way_decision(
                nsfw_score, self.safe_threshold, self.nsfw_threshold
            )
            predicted_labels = [
                label for label, score in risk_scores.items()
                if score >= self.safe_threshold
            ]
            predicted_labels = list(dict.fromkeys(predicted_labels + sorted(secondary_predicted)))
            label_scores = [
                ImageLabelScore(
                    label=label,
                    score=score,
                    predicted=label in predicted_labels,
                )
                for label, score in sorted(
                    risk_scores.items(), key=lambda item: item[1], reverse=True
                )
            ]
            top_label = label_scores[0].label if label_scores else "no_risk_detected"
            results.append(
                ImageModerationResult(
                    source=source,
                    labels=label_scores,
                    predicted_labels=predicted_labels,
                    top_label=top_label,
                    top_score=nsfw_score,
                    model_version=self.model_version,
                    decision=decision,
                    nsfw_score=nsfw_score,
                    method=(
                        "supervised-pretrained-inference+secondary-clip"
                        if self.secondary_clip
                        else "supervised-pretrained-inference"
                    ),
                    categories=categories,
                )
            )
        return results

    def _predict_zero_shot_ensemble(
        self, image_list: list[ImageInput], sources: list[str]
    ) -> list[ImageModerationResult]:
        """
        Zero-shot prediction with prompt ensembling.

        For each image:
          1. Encode image → normalized embedding
          2. For each category: average similarity over all positive/negative prompts
          3. Compute NSFW ensemble score: average similarity to all NSFW prompts
          4. Compute SAFE ensemble score: average similarity to all safe prompts
          5. Softmax → NSFW pseudo-probability
          6. Three-way decision: SAFE / REVIEW / NSFW
        """
        image_features = torch.from_numpy(self.encode_images(image_list)).to(self.device)
        logit_scale = self.model.logit_scale.exp() if hasattr(self.model, "logit_scale") else 100.0

        results: list[ImageModerationResult] = []
        for i, source in enumerate(sources):
            feat = image_features[i]  # (D,)

            # --- Per-label scoring (with ensembled prompts) ---
            label_scores: list[ImageLabelScore] = []
            for label_idx, label in enumerate(self.labels):
                pos_text, neg_text = self._category_text_features[label_idx]
                sims = torch.stack([feat @ pos_text, feat @ neg_text]) * logit_scale
                probs = torch.softmax(sims, dim=0)
                pos_prob = float(probs[0])
                label_scores.append(
                    ImageLabelScore(
                        label=label,
                        score=pos_prob,
                        predicted=pos_prob >= self.label_decision_cutoff,
                    )
                )

            # --- NSFW ensemble scoring (prompt ensembling) ---
            nsfw_sims = self._nsfw_text_features @ feat * logit_scale  # (N_nsfw,)
            safe_sims = self._safe_text_features @ feat * logit_scale  # (N_safe,)

            # Average similarity across all prompts in each class
            nsfw_avg = float(nsfw_sims.mean())
            safe_avg = float(safe_sims.mean())

            # Softmax → NSFW pseudo-probability
            nsfw_score = _softmax_similarity(nsfw_avg, safe_avg)

            # --- Three-way routing ---
            decision = _three_way_decision(
                nsfw_score, self.safe_threshold, self.nsfw_threshold
            )

            top = max(label_scores, key=lambda ls: ls.score)
            results.append(
                ImageModerationResult(
                    source=source,
                    labels=label_scores,
                    predicted_labels=[ls.label for ls in label_scores if ls.predicted],
                    top_label=top.label,
                    top_score=top.score,
                    model_version=self.model_version,
                    nsfw_score=nsfw_score,
                    decision=decision,
                    method="zero-shot-clip-prompt-ensemble",
                )
            )

            logger.debug(
                "Image %s: nsfw_score=%.4f, safe_avg=%.4f, nsfw_avg=%.4f → %s",
                source, nsfw_score, safe_avg, nsfw_avg, decision,
            )

        return results

    def _predict_with_linear_probe(
        self, image_list: list[ImageInput], sources: list[str]
    ) -> list[ImageModerationResult]:
        """Prediction using a trained linear probe on frozen CLIP embeddings."""
        embeddings = self.encode_images(image_list)
        probs = self.linear_probe.predict_proba(embeddings)

        results: list[ImageModerationResult] = []
        for i, source in enumerate(sources):
            row = probs[i]
            label_scores = [
                ImageLabelScore(
                    label=label,
                    score=float(score),
                    predicted=bool(score >= self.label_decision_cutoff),
                )
                for label, score in zip(self.labels, row)
            ]
            top = max(label_scores, key=lambda ls: ls.score)

            # For linear probe path, use the nsfw label score as nsfw_score
            nsfw_label_score = 0.0
            for ls in label_scores:
                if ls.label == "nsfw":
                    nsfw_label_score = ls.score
                    break
            decision = _three_way_decision(
                nsfw_label_score, self.safe_threshold, self.nsfw_threshold
            )

            results.append(
                ImageModerationResult(
                    source=source,
                    labels=label_scores,
                    predicted_labels=[ls.label for ls in label_scores if ls.predicted],
                    top_label=top.label,
                    top_score=top.score,
                    model_version=self.model_version,
                    nsfw_score=nsfw_label_score,
                    decision=decision,
                    method="linear-probe",
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Persistence (used by retrain.py)
    # ------------------------------------------------------------------ #
    def save_pretrained(self, output_dir: str | Path) -> None:
        """Save the CLIP state_dict to output_dir/clip_finetuned.pt."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / "clip_finetuned.pt")
        logger.info("Saved image moderation model to %s", output_dir / "clip_finetuned.pt")

    @staticmethod
    def save_linear_probe(probe, output_dir: str | Path) -> Path:
        """Persist a trained sklearn multi-label classifier."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_path = output_dir / "linear_probe.joblib"
        joblib.dump(probe, probe_path)
        logger.info("Saved linear probe classifier to %s", probe_path)
        return probe_path


# --------------------------------------------------------------------------- #
# Singleton (for FastAPI dependency injection)
# --------------------------------------------------------------------------- #
_singleton: ImageModerationModel | None = None


def get_image_moderation_model() -> ImageModerationModel:
    """Process-wide singleton accessor, injected by FastAPI."""
    global _singleton
    if _singleton is None:
        _singleton = ImageModerationModel.from_pretrained()
    return _singleton


def reload_image_moderation_model() -> ImageModerationModel:
    """Force the singleton to reload from disk (e.g. after retrain.py)."""
    global _singleton
    _singleton = ImageModerationModel.from_pretrained()
    logger.info("Image moderation model reloaded: version=%s", _singleton.model_version)
    return _singleton
