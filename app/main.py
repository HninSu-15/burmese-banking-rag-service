# app/main.py
"""FastAPI entrypoint for the Burmese Banking RAG Service."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.v1.routes import router as v1_router
from app.services.rag.factory import RAGServiceFactory
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm lazily-initialized RAG components on startup."""
    logger.info("Starting %s v%s", settings.SERVICE_NAME, settings.SERVICE_VERSION)
    try:
        RAGServiceFactory.initialize()
        logger.info("RAG Service Factory initialized successfully")
    except Exception as exc:
        # Do not crash the app; /health reports the degraded state.
        logger.error("RAG Service Factory initialization failed: %s", exc)
    yield
    logger.info("Shutting down %s", settings.SERVICE_NAME)


app = FastAPI(
    title="Burmese Banking RAG Service",
    description=(
        "Grounded knowledge retrieval layer for the AI-Powered Burmese Voice Support Copilot. "
        "POST /api/v1/retrieve returns the exact JSON context contract consumed by the LLM team."
    ),
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

app.include_router(v1_router, prefix="/api/v1", tags=["rag"])