# SERS Cancer Screening — Production Inference Image
# Lightweight image for running predictions on new spectra.
#
# Build:
#   docker build -f infra/Dockerfile -t sers-predict .
#
# Run:
#   docker run --rm -v /path/to/spectra:/spectra sers-predict /spectra/sample.CSV
#   docker run --rm -v /path/to/spectra:/spectra sers-predict /spectra/sample.CSV --age 65 --sex M --bmi 24.3
#   docker run --rm -v /path/to/spectra:/spectra sers-predict /spectra/*.CSV --quiet --output /spectra/results.json

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install only runtime dependencies (no torch, no GPU)
COPY pyproject.toml pyproject.toml
COPY src/ src/
RUN pip install --no-cache-dir "."

# Also need joblib and matplotlib for the predictor
RUN pip install --no-cache-dir joblib matplotlib


# ── Stage 2: Runtime ──
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only what's needed for inference
COPY src/ src/
COPY config/config.yaml config/config.yaml
COPY scripts/deployment/sers_predict.py scripts/deployment/sers_predict.py
COPY artifacts/usersnet/ artifacts/usersnet/
COPY artifacts/baselines/ artifacts/baselines/

# Non-root user
RUN useradd -m -u 1000 sers && chown -R sers:sers /app
USER sers

# Health check
HEALTHCHECK --interval=60s --timeout=5s --retries=2 \
    CMD python -c "import json; json.load(open('artifacts/usersnet/current/manifest.json'))" || exit 1

ENTRYPOINT ["python", "scripts/deployment/sers_predict.py"]
