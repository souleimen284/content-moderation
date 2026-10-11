"""
FastAPI dependency accessors.

Thin wrappers around the singletons/services from earlier modules, so
route handlers depend on these (mockable/overridable in tests via
``app.dependency_overrides``) rather than importing singletons directly.
"""

from __future__ import annotations

from app.db.database import get_db  # noqa: F401  (re-exported for convenience)
from app.ml.image_moderation import ImageModerationModel, get_image_moderation_model
from app.ml.text_moderation import TextModerationModel, get_text_moderation_model
from app.services.routing import ConfidenceRouter, get_confidence_router


def get_text_model() -> TextModerationModel:
    return get_text_moderation_model()


def get_image_model() -> ImageModerationModel:
    return get_image_moderation_model()


def get_router_service() -> ConfidenceRouter:
    return get_confidence_router()
