"""
POST /moderate/text, POST /moderate/image, POST /moderate/content

Each endpoint runs the full pipeline: create a ModerationRequest, run the
relevant model(s) (Modules 2-3), route the result through the confidence
policy (Module 4), persist everything (Module 5), and return the
prediction + routing decision to the caller.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_image_model, get_router_service, get_text_model
from app.core.logging_config import get_logger
from app.db.models import ContentType, Modality
from app.ml.image_moderation import ImageModerationModel
from app.ml.text_moderation import TextModerationModel
from app.schemas.api import ModerateContentResponse, ModerateImageResponse, ModerateTextResponse
from app.schemas.text_moderation import TextModerationRequest
from app.services.queue_service import (
    apply_routing_decision,
    create_moderation_request,
    record_prediction,
)
from app.services.routing import ConfidenceRouter
from app.utils.file_storage import save_upload_file

logger = get_logger(__name__)
router = APIRouter(tags=["moderation"])


@router.post("/moderate/text", response_model=ModerateTextResponse)
def moderate_text(
    payload: TextModerationRequest,
    db: Session = Depends(get_db),
    text_model: TextModerationModel = Depends(get_text_model),
    confidence_router: ConfidenceRouter = Depends(get_router_service),
) -> ModerateTextResponse:
    """Moderate a single piece of text: predict, route, persist, respond."""
    request = create_moderation_request(
        db,
        content_type=ContentType.TEXT,
        text_content=payload.text,
        request_metadata=payload.metadata,
    )

    result = text_model.predict(payload.text)[0]
    decision = confidence_router.route_text(result)

    record_prediction(db, request, Modality.TEXT, result, decision.tier.value, decision.action.value)
    request = apply_routing_decision(db, request, decision)

    return ModerateTextResponse(
        request_id=request.id, status=request.status.value, prediction=result, routing=decision
    )


@router.post("/moderate/image", response_model=ModerateImageResponse)
async def moderate_image(
    image: UploadFile = File(..., description="Image file to moderate."),
    db: Session = Depends(get_db),
    image_model: ImageModerationModel = Depends(get_image_model),
    confidence_router: ConfidenceRouter = Depends(get_router_service),
) -> ModerateImageResponse:
    """Moderate a single uploaded image: predict, route, persist, respond."""
    try:
        image_path = await save_upload_file(image)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    request = create_moderation_request(db, content_type=ContentType.IMAGE, image_path=image_path)

    try:
        result = image_model.predict(image_path, sources=[image.filename or "upload"])[0]
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Rejected invalid image upload %s: %s", image.filename, exc)
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc
    decision = confidence_router.route_image(result)

    record_prediction(db, request, Modality.IMAGE, result, decision.tier.value, decision.action.value)
    request = apply_routing_decision(db, request, decision)

    return ModerateImageResponse(
        request_id=request.id, status=request.status.value, prediction=result, routing=decision
    )


@router.post("/moderate/content", response_model=ModerateContentResponse)
async def moderate_content(
    text: str | None = Form(default=None, description="Text content, e.g. a caption."),
    image: UploadFile | None = File(default=None, description="Optional image file."),
    db: Session = Depends(get_db),
    text_model: TextModerationModel = Depends(get_text_model),
    image_model: ImageModerationModel = Depends(get_image_model),
    confidence_router: ConfidenceRouter = Depends(get_router_service),
) -> ModerateContentResponse:
    """
    Moderate a piece of content that may include text, an image, or both
    (e.g. a social post with a caption and a photo). Routes via the
    combined multi-modal policy (Module 4).
    """
    has_text = text is not None and text.strip() != ""
    has_image = image is not None and image.filename

    if not has_text and not has_image:
        raise HTTPException(status_code=422, detail="At least one of 'text' or 'image' is required.")

    content_type = ContentType.COMBINED if (has_text and has_image) else (
        ContentType.TEXT if has_text else ContentType.IMAGE
    )

    image_path: str | None = None
    if has_image:
        try:
            image_path = await save_upload_file(image)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

    request = create_moderation_request(
        db,
        content_type=content_type,
        text_content=text.strip() if has_text else None,
        image_path=image_path,
    )

    text_result = text_model.predict(text.strip())[0] if has_text else None
    try:
        image_result = (
            image_model.predict(image_path, sources=[image.filename or "upload"])[0]
            if has_image
            else None
        )
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Rejected invalid image upload %s: %s", image.filename, exc)
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    decision = confidence_router.route_content(text_result=text_result, image_result=image_result)

    if text_result is not None:
        record_prediction(db, request, Modality.TEXT, text_result, decision.tier.value, decision.action.value)
    if image_result is not None:
        record_prediction(db, request, Modality.IMAGE, image_result, decision.tier.value, decision.action.value)
    request = apply_routing_decision(db, request, decision)

    return ModerateContentResponse(
        request_id=request.id,
        status=request.status.value,
        text_prediction=text_result,
        image_prediction=image_result,
        routing=decision,
    )
