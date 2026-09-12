#!/bin/bash
# scripts/run.sh
# Run the Docker container

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

IMAGE_NAME="health-predictor"
VERSION="${1:-latest}"
PORT="${PORT:-8000}"

echo -e "${GREEN}Running container: ${IMAGE_NAME}:${VERSION} on port ${PORT}${NC}"

docker run -p ${PORT}:8000 \
    -e LOG_LEVEL=INFO \
    -e MODEL_PATH=/app/models/chronic_disease_predictor_v1.0.joblib \
    -v $(pwd)/models:/app/models \
    --name health-predictor-container \
    ${IMAGE_NAME}:${VERSION}