# ============================================================
# DOCKERFILE FOR CHRONIC DISEASE PREDICTOR API
# ============================================================
#
# This Dockerfile creates a container that runs the FastAPI
# application serving the trained model.
#
# Build: docker build -t health-predictor:v1.0 .
# Run:   docker run -p 8000:8000 health-predictor:v1.0
#
# ============================================================

# ============================================================
# STAGE 1: Builder (for dependencies)
# ============================================================
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# STAGE 2: Final image
# ============================================================
FROM python:3.12-slim

# Metadata
LABEL maintainer="noco0226@gmail.com"
LABEL version="1.0.0"
LABEL description="Chronic Disease Predictor API"
LABEL org.opencontainers.image.source="https://github.com/ackben0226/chronic-disease-predictor"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=4 \
    MODEL_PATH=/app/models/chronic_disease_predictor_v1.0.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY deployment/ ./deployment/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Copy package configuration
COPY pyproject.toml .
COPY requirements.txt .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "deployment.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]