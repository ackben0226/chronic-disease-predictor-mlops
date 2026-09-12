#!/bin/bash
# scripts/build.sh
# Build the Docker image

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
IMAGE_NAME="health-predictor"
VERSION="${1:-latest}"
REGISTRY="${REGISTRY:-}"

# Build image
echo -e "${GREEN}Building Docker image: ${IMAGE_NAME}:${VERSION}${NC}"

if [ -n "$REGISTRY" ]; then
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker build -t "${IMAGE_NAME}:${VERSION}" -t "${FULL_IMAGE}" -f Dockerfile .
    echo -e "${GREEN}✅ Image built: ${FULL_IMAGE}${NC}"
else
    docker build -t "${IMAGE_NAME}:${VERSION}" -f Dockerfile .
    echo -e "${GREEN}✅ Image built: ${IMAGE_NAME}:${VERSION}${NC}"
fi

# Show image details
docker images | grep ${IMAGE_NAME}