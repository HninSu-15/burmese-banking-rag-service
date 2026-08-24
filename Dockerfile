# Dockerfile for Burmese Banking RAG Service
# Multi-stage build:
#   Stage "builder": compile llama-cpp-python + install deps (as root)
#   Stage "runtime": slim image with deps copied over + model pre-downloaded

# ---------------------------------------------------------------------------
# Builder stage — heavy build tools live here only
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# Install as ROOT so packages land in /usr/local (easy to copy out later)
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime stage — slim, non-root, model pre-downloaded
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Python packages + entrypoint scripts from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user AND app directories BEFORE switching user
RUN useradd -m -s /bin/bash appuser \
    && mkdir -p /app/models /app/.cache/huggingface /app/data/chroma_db /app/logs \
    && chown -R appuser:appuser /app

WORKDIR /app
USER appuser

# HuggingFace cache inside the image — pre-downloaded model lives here,
# matching exactly what app/services/rag/embedder_service.py uses at runtime
# (repo_id='Qwen/Qwen3-Embedding-0.6B-GGUF', filename='Qwen3-Embedding-0.6B-Q8_0.gguf')
ENV HF_HOME=/app/.cache/huggingface
ENV MODEL_DIR=/app/models

# Pre-download the Qwen3 GGUF model during build (baked into image).
# Same repo_id/filename/cache as runtime => hf_hub_download finds it instantly.
RUN python -c " \
    from huggingface_hub import hf_hub_download; \
    hf_hub_download( \
        repo_id='Qwen/Qwen3-Embedding-0.6B-GGUF', \
        filename='Qwen3-Embedding-0.6B-Q8_0.gguf' \
    ) \
" || echo "WARNING: Model download failed during build. Will download at runtime."

# Copy application source
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser core/ ./core/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser knowledge/ ./knowledge/
COPY --chown=appuser:appuser requirements.txt ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]