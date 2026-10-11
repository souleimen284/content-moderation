"""
Local file storage for uploaded images.

Uploaded files are saved under ``settings.upload_dir`` with a
collision-proof generated filename, preserving the original extension.
The ``ModerationRequest.image_path`` column stores this path, which is
what ``ImageModerationModel.predict`` and the retraining pipeline
(``export_moderator_reviewed_images_for_retraining``) read from later.

In a multi-instance deployment, ``upload_dir`` should point at a shared
volume or object storage mount (see docker-compose.yml, Module 9) rather
than local container disk.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _safe_extension(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in ALLOWED_IMAGE_EXTENSIONS else ".jpg"


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Persist an uploaded image to disk and return its path as a string.

    Raises:
        ValueError: if the file exceeds ``settings.max_upload_size_mb``.
    """
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    ext = _safe_extension(upload_file.filename)
    dest_path = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    size = 0
    with open(dest_path, "wb") as f:
        while chunk := await upload_file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Uploaded file exceeds the {settings.max_upload_size_mb}MB size limit"
                )
            f.write(chunk)

    await upload_file.seek(0)
    logger.info("Saved upload %s (%d bytes) to %s", upload_file.filename, size, dest_path)
    return str(dest_path)
