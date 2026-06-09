# ============================================================
# Stage 1 — Build the React frontend
# ============================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Install dependencies first (cached layer unless package.json changes).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --frozen-lockfile

COPY frontend/ ./

# Build with an empty API base so every fetch uses a relative URL
# (same origin as the backend). The ?? fallback in api.ts only fires
# for undefined, not for the empty string, so this works correctly.
RUN VITE_API_BASE="" npm run build


# ============================================================
# Stage 2 — Python backend + pre-built frontend
# ============================================================
FROM python:3.11-slim

# Non-root user for least-privilege operation.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install system dependencies needed by scipy/sklearn at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python package and its Anthropic extra so a real LLM is
# available when ANTHROPIC_API_KEY is set at runtime.
# aiofiles is required by FastAPI's StaticFiles middleware.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[anthropic]" \
    && pip install --no-cache-dir aiofiles

# MEPS training data — baked into the image so the model can train
# inside the container without needing an external data mount.
COPY data/ ./data/

# Built React SPA — served as static files by FastAPI.
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Persistent data lives on a volume so it survives container restarts:
#   /data/gauge.db  — SQLite database (sessions + documents)
#   /data/cache/         — trained model cache (.joblib)
ENV GAUGE_DB_PATH=/data/gauge.db
ENV GAUGE_CACHE_DIR=/data/cache

RUN mkdir -p /data && chown appuser:appuser /data

USER appuser

EXPOSE 8000

# Serve with a single worker; the in-process SQLite store is not safe
# across multiple worker processes. Use --workers 1 explicitly.
CMD ["uvicorn", "gauge.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
