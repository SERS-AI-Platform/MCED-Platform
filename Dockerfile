# SERS Analysis Pipeline - Production Multi-Stage Build
# Stage 1: Builder - install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (only needed in builder stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency files and source (needed for package discovery)
COPY pyproject.toml pyproject.toml
COPY src/ src/

# Install core + optional dependencies
RUN pip install --no-cache-dir ".[ml,viz]"


# Stage 2: Runtime - minimal final image
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment for venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERS_DATA_DIR=/data \
    SERS_RESULTS_DIR=/results \
    SERS_MODEL_DIR=/models

# Copy application code
COPY src/ src/
COPY config.yaml config.yaml.example
COPY main.py main.py

# Create required directories
RUN mkdir -p /data /results /models && \
    chmod 755 /data /results /models

# Health check (basic validation)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.sers.config import load_config; load_config('config.yaml.example')" || exit 1

# Non-root user (security best practice)
RUN useradd -m -u 1000 sers && \
    chown -R sers:sers /app /data /results /models
USER sers

# Entry point with logging
ENTRYPOINT ["python", "main.py"]
