"""
Database engine and session management.

Uses ``settings.database_url`` (PostgreSQL in staging/production, per
``.env``). SQLite is also supported transparently for local development
and the test suite (Module 10) by pointing ``DATABASE_URL`` at
``sqlite:///...`` — every model in ``app.db.models`` uses portable
SQLAlchemy types so the schema behaves identically on both backends.

In production, schema changes should go through Alembic migrations
(``alembic/`` — see the ``xlsx``... err, ``docs``/README for setup). For
local development and tests, ``init_db()`` provides a fast
``create_all()`` path.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


def _make_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        # Needed for SQLite to be usable across threads (FastAPI request
        # handlers, background tasks) with a single file/in-memory DB.
        connect_args = {"check_same_thread": False}

    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
        future=True,
    )


engine = _make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """
    Create all tables that don't yet exist.

    Safe to call multiple times (no-op for existing tables). Used at app
    startup in development and in the test suite; production deployments
    should prefer Alembic migrations for schema evolution.
    """
    # Import models so they're registered on Base.metadata before create_all.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured at %s", _redact(settings.database_url))


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a request-scoped DB session, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for use outside of FastAPI (scripts, retrain.py, tests):

        with session_scope() as db:
            db.add(obj)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _redact(url: str) -> str:
    """Hide credentials when logging a DB URL."""
    if "@" not in url:
        return url
    scheme_and_creds, rest = url.split("@", 1)
    scheme = scheme_and_creds.split("://", 1)[0]
    return f"{scheme}://***:***@{rest}"
