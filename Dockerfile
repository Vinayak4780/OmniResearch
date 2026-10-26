# syntax=docker/dockerfile:1.7
# OmniResearch: one image serving the API and the built React UI.
#   docker build -t omniresearch .                      # with Docling + PaddleOCR + fastembed (CPU)
#   docker build -t omniresearch --build-arg INSTALL_ML=false .   # slim image, PyMuPDF parser, no local models

FROM node:22-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime
ARG INSTALL_ML=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    OMNI_DATA_DIR=/data \
    HF_HOME=/models/huggingface \
    PADDLE_PDX_CACHE_HOME=/models/paddlex \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Shared libraries needed by OpenCV/PaddleOCR/Docling on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 omni
WORKDIR /app/backend
COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_ML" = "true" ]; then pip install -r requirements-ml.txt; fi

COPY backend/app ./app
COPY --from=frontend /src/frontend/dist /app/frontend/dist

RUN mkdir -p /data /models && chown -R omni:omni /data /models /app
USER omni
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
