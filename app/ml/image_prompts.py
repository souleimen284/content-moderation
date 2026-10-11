"""
Prompt ensemble configuration for zero-shot CLIP image moderation.

This module defines the prompt sets used for prompt ensembling — a technique
where multiple descriptive prompts per category are embedded and averaged to
produce a more robust similarity score than any single prompt.

Design principles:
  - Multiple NSFW prompts capture different types of explicit content
  - Multiple SAFE prompts describe normal, non-explicit images
  - False-positive mitigation prompts handle common CLIP confusion areas
    (beaches, swimsuits, fitness, medical, artistic photography)
  - All prompts are simple English phrases that CLIP understands well

NOTE: These are zero-shot prompts for a pretrained CLIP model. They are NOT
calibrated probabilities. The softmax scoring converts raw similarities into
a pseudo-probability that should be interpreted as a confidence score, not a
true probability of NSFW content.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# NSFW prompts — explicit sexual content
# --------------------------------------------------------------------------- #
DEFAULT_NSFW_PROMPTS: list[str] = [
    "a sexually explicit image",
    "a pornographic image",
    "an image containing explicit sexual content",
    "an image showing explicit nudity in a sexual context",
    "an image showing sexual activity",
    "an adult image with explicit sexual content",
    "an image depicting hardcore pornography",
    "a photograph of explicit adult content",
]


# --------------------------------------------------------------------------- #
# SAFE prompts — normal, non-explicit content
# --------------------------------------------------------------------------- #
DEFAULT_SAFE_PROMPTS: list[str] = [
    "a normal safe-for-work photograph",
    "a non-explicit photograph",
    "a family-friendly photograph",
    "an ordinary photograph without nudity",
    "a safe-for-work image",
    "a normal photograph with no explicit sexual content",
    "a wholesome photograph appropriate for all audiences",
    "a clean, appropriate photograph",
]


# --------------------------------------------------------------------------- #
# False-positive mitigation prompts
# --------------------------------------------------------------------------- #
# These prompts describe contexts that CLIP frequently confuses with NSFW
# content. They are used as additional SAFE evidence — if the image is
# similar to these prompts, the SAFE score is boosted.
DEFAULT_FP_MITIGATION_PROMPTS: list[str] = [
    "a person wearing a swimsuit at a beach",
    "a fitness photograph of someone exercising",
    "a normal medical photograph in a clinical setting",
    "a fashion photograph showing clothing",
    "an artistic photograph or painting",
    "a normal portrait of a person",
    "a person at a swimming pool wearing swimwear",
    "a yoga photograph showing athletic poses",
    "a figure drawing or art class sketch",
    "a beach vacation photograph",
]


# --------------------------------------------------------------------------- #
# Per-category prompts (for multi-label scoring)
# --------------------------------------------------------------------------- #
# Each category has a set of positive prompts that describe the violation
# and negative prompts that describe the absence of the violation.
DEFAULT_CATEGORY_PROMPTS: dict[str, dict[str, list[str]]] = {
    "nsfw": {
        "positive": DEFAULT_NSFW_PROMPTS,
        "negative": DEFAULT_SAFE_PROMPTS,
    },
    "violence": {
        "positive": [
            "a photograph depicting physical violence",
            "an image showing someone being harmed or attacked",
            "a violent scene with fighting or assault",
            "an image of a physical altercation or brawl",
            "a photograph showing graphic violence or injury",
        ],
        "negative": [
            "a peaceful photograph",
            "a calm scene with no violence",
            "a normal photograph without any aggression",
        ],
    },
    "weapons": {
        "positive": [
            "a photograph showing firearms or weapons",
            "an image containing guns, knives, or other weapons",
            "a threatening image with visible weapons",
            "a photograph of someone holding a weapon",
        ],
        "negative": [
            "a photograph without any weapons",
            "a safe scene with no weapons visible",
            "a normal photograph of everyday objects",
        ],
    },
    "hate_symbols": {
        "positive": [
            "an image containing hate symbols or extremist imagery",
            "a photograph with nazi symbols or hate group imagery",
            "an image displaying racist or hateful symbols",
        ],
        "negative": [
            "a photograph without any hate symbols",
            "a normal photograph with neutral imagery",
            "a safe image with no extremist content",
        ],
    },
    "graphic_content": {
        "positive": [
            "a photograph showing graphic or disturbing content",
            "an image depicting gore or severe injury",
            "a disturbing photograph with graphic visual content",
            "an image showing blood and injury",
        ],
        "negative": [
            "a photograph without any graphic content",
            "a safe image with nothing disturbing",
            "a normal clean photograph",
        ],
    },
}


@dataclass
class PromptEnsembleConfig:
    """Configuration for the prompt ensemble system."""

    nsfw_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_NSFW_PROMPTS))
    safe_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_SAFE_PROMPTS))
    fp_mitigation_prompts: list[str] = field(
        default_factory=lambda: list(DEFAULT_FP_MITIGATION_PROMPTS)
    )
    category_prompts: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: {k: {sk: list(sv) for sk, sv in v.items()} for k, v in DEFAULT_CATEGORY_PROMPTS.items()}
    )
    use_fp_mitigation: bool = True

    def get_all_safe_prompts(self) -> list[str]:
        """Return SAFE prompts + false-positive mitigation prompts."""
        prompts = list(self.safe_prompts)
        if self.use_fp_mitigation:
            prompts.extend(self.fp_mitigation_prompts)
        return prompts

    def get_nsfw_prompts(self) -> list[str]:
        return list(self.nsfw_prompts)


def load_prompt_config(raw: dict | None = None) -> PromptEnsembleConfig:
    """
    Build a PromptEnsembleConfig from a raw YAML dict (from model_config.yaml).
    Falls back to defaults for any missing keys.
    """
    if raw is None:
        return PromptEnsembleConfig()

    nsfw = raw.get("nsfw_prompts")
    safe = raw.get("safe_prompts")
    fp = raw.get("fp_mitigation_prompts")
    use_fp = raw.get("use_fp_mitigation", True)

    cat_raw = raw.get("category_prompts")
    cat_prompts: dict[str, dict[str, list[str]]] | None = None
    if cat_raw:
        cat_prompts = {}
        for label, prompts_dict in cat_raw.items():
            cat_prompts[label] = {
                "positive": prompts_dict.get("positive", []),
                "negative": prompts_dict.get("negative", []),
            }

    return PromptEnsembleConfig(
        nsfw_prompts=nsfw if nsfw else list(DEFAULT_NSFW_PROMPTS),
        safe_prompts=safe if safe else list(DEFAULT_SAFE_PROMPTS),
        fp_mitigation_prompts=fp if fp else list(DEFAULT_FP_MITIGATION_PROMPTS),
        category_prompts=cat_prompts if cat_prompts else {
            k: {sk: list(sv) for sk, sv in v.items()}
            for k, v in DEFAULT_CATEGORY_PROMPTS.items()
        },
        use_fp_mitigation=use_fp,
    )
