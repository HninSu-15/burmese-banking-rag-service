# app/api/deps.py
"""
Shared FastAPI dependencies: internal API token authentication.
"""

from fastapi import Header, HTTPException, status

from core.config import settings

# Header name expected from the gateway when calling this RAG service.
API_TOKEN_HEADER = "X-RAG-Service-Token"


def require_service_token(
    x_rag_service_token: str = Header(default="", alias=API_TOKEN_HEADER),
) -> None:
    """Reject requests without a valid internal service token.

    When ``settings.RAG_API_TOKEN`` is empty, auth is disabled
    (development mode). Configure ``RAG_API_TOKEN`` in ``.env``
    before exposing the service to other environments.
    """
    if not settings.RAG_API_TOKEN:
        return
    if not x_rag_service_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing RAG service token",
        )
    if x_rag_service_token != settings.RAG_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid RAG service token",
        )