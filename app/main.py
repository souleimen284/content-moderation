"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI is automatically available at /docs, ReDoc at /redoc.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import moderation, queue, system
from app.core.config import settings
from app.core.logging_config import configure_logging, get_logger
from app.db.database import init_db
from app.ml.image_moderation import get_image_moderation_model

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)
    init_db()
    # Load CLIP before accepting requests so the first image upload does not
    # block behind the frontend proxy timeout.
    get_image_moderation_model()
    logger.info("Image moderation model ready")
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Production content moderation pipeline: multi-label text moderation "
        "(fine-tuned BERT), image moderation (CLIP), confidence-based routing, "
        "human-in-the-loop review, and weekly automated retraining."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(moderation.router, prefix=settings.api_v1_prefix)
app.include_router(queue.router, prefix=settings.api_v1_prefix)
app.include_router(system.router, prefix=settings.api_v1_prefix)

# Also expose /health and /metrics unprefixed, since load balancers and
# orchestrators (Docker/Kubernetes health checks) conventionally probe
# these paths without an API version prefix.
app.include_router(system.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
