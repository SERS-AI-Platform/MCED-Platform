# SERS Analysis Pipeline
# Multi-stage build for smaller final image

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .


# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy config template
COPY config.yaml config.yaml.example

# Create directories for data mounting
RUN mkdir -p /data /results /models

# Environment variables for paths
ENV SERS_DATA_DIR=/data
ENV SERS_RESULTS_DIR=/results
ENV SERS_MODEL_DIR=/models

# Default command
ENTRYPOINT ["sers"]
CMD ["--help"]

# Usage:
# docker build -t sers-analysis .
# docker run -v $(pwd)/data:/data -v $(pwd)/results:/results sers-analysis run
