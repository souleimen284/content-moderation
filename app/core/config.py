"""
Centralized application configuration.

All runtime configuration is defined here as a single Pydantic ``Settings``
object, populated from environment variables (and a local ``.env`` file in
development). No other module should read ``os.environ`` directly — import
``settings`` from this module instead, so that configuration stays
consistent, typed, and testable (settings can be overridden in tests via
``Settings(**overrides)``).

Confidence-routing thresholds, model names, database URL, and MLflow config
all live here because they change per-environment (dev/staging/prod) but
must never require a code change to adjust.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = Field(default="Content Moderation Pipeline")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=True)
    api_v1_prefix: str = Field(default="/api/v1")
    secret_key: str = Field(default="dev-secret-change-me")
    access_token_expire_minutes: int = Field(default=60)

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Database ---
    postgres_user: str = Field(default="moderation_user")
    postgres_password: str = Field(default="change-me")
    postgres_db: str = Field(default="content_moderation")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    database_url: str | None = Field(default=None)

    # --- ML: Text model ---
    text_model_name: str = Field(default="distilbert-base-uncased")
    text_model_checkpoint_dir: Path = Field(default=Path("./models/text_moderation"))
    text_model_max_length: int = Field(default=128)
    text_model_freeze_encoder: bool = Field(default=True)
    text_model_freeze_layers: int = Field(default=4)
    text_model_max_train_samples: int = Field(default=50000)
    text_label_decision_cutoff: float = Field(default=0.70, ge=0.0, le=1.0)
    text_labels: list[str] = Field(
        default_factory=lambda: ["toxic", "spam", "hate"]
    )

    # --- ML: Image model ---
    image_model_name: str = Field(default="ViT-L-14")
    image_model_pretrained: str = Field(default="openai")
    image_model_checkpoint_dir: Path = Field(default=Path("./models/image_moderation"))
    image_model_provider: Literal["nudenet", "clip", "ensemble"] = Field(default="nudenet")
    image_safe_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    image_nsfw_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    image_clip_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    image_labels: list[str] = Field(
        default_factory=lambda: [
            "nsfw",
            "violence",
            "weapons",
            "hate_symbols",
            "graphic_content",
        ]
    )

    # --- Image moderation thresholds (three-way routing) ---
    # These control the SAFE / REVIEW / NSFW decision.
    # NOTE: These are pseudo-probability thresholds, not true calibrated
    # probabilities. They should be tuned using a validation dataset.
    image_safe_threshold: float = Field(default=0.30, ge=0.0, le=1.0,
        description="NSFW score below this → SAFE")
    image_nsfw_threshold: float = Field(default=0.70, ge=0.0, le=1.0,
        description="NSFW score above this → NSFW; between safe and nsfw → REVIEW")

    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")

    # --- File uploads ---
    upload_dir: Path = Field(default=Path("./data/uploads"))
    max_upload_size_mb: int = Field(default=10)

    # --- Confidence-based routing thresholds ---
    auto_decision_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    queue_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    # --- MLflow ---
    use_mlflow: bool = Field(default=False)
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="content-moderation")

    # --- Retraining ---
    retrain_schedule_cron: str = Field(default="0 3 * * 1")
    retrain_min_new_samples: int = Field(default=100)
    retrain_data_dir: Path = Field(default=Path("./data/retraining"))

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)
    log_dir: Path = Field(default=Path("./logs"))

    @field_validator("queue_threshold")
    @classmethod
    def _queue_below_auto(cls, v: float, info) -> float:
        """Ensure queue_threshold never exceeds auto_decision_threshold."""
        auto = info.data.get("auto_decision_threshold")
        if auto is not None and v > auto:
            raise ValueError(
                f"queue_threshold ({v}) cannot exceed auto_decision_threshold ({auto})"
            )
        return v

    @model_validator(mode="after")
    def _assemble_database_url(self) -> "Settings":
        """Build DATABASE_URL from parts if not explicitly provided."""
        if not self.database_url:
            self.database_url = (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def classify_confidence(self, confidence: float) -> Literal["auto", "queue", "uncertain"]:
        """
        Map a raw confidence score to a routing decision per the
        configurable three-tier policy:

          - confidence > auto_decision_threshold  -> "auto"      (auto approve/reject)
          - queue_threshold <= confidence <= auto_decision_threshold -> "queue" (moderator queue)
          - confidence < queue_threshold           -> "uncertain" (manual review)
        """
        if confidence > self.auto_decision_threshold:
            return "auto"
        if confidence >= self.queue_threshold:
            return "queue"
        return "uncertain"


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached ``Settings`` singleton.

    Using ``lru_cache`` means the environment is read once per process,
    which is fast and predictable. Tests that need different settings
    should call ``get_settings.cache_clear()`` after monkeypatching env
    vars, or construct ``Settings(...)`` directly with overrides.
    """
    return Settings()


settings = get_settings()
