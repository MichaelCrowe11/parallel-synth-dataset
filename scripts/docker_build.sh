#!/bin/bash
# Build and Push Parallel Synth Docker Images

set -e

# Configuration
DOCKER_USERNAME=${DOCKER_USERNAME:-"parallelsynth"}
IMAGE_NAME="dataset-generator"
VERSION=${VERSION:-"latest"}
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

echo "=================================================="
echo "  Building Parallel Synth Docker Image"
echo "=================================================="
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Build the image
echo "Building Docker image..."
echo ""

docker build \
    -t ${FULL_IMAGE} \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:latest \
    --build-arg BUILDDATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VERSION=${VERSION} \
    .

echo ""
echo "✓ Build complete!"
echo ""

# Show image info
docker images ${DOCKER_USERNAME}/${IMAGE_NAME}

echo ""
echo "=================================================="
echo "  Image built successfully!"
echo "=================================================="
echo ""
echo "Image tags:"
echo "  - ${FULL_IMAGE}"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "Test locally:"
echo "  docker run --rm ${FULL_IMAGE}"
echo ""
echo "Push to registry:"
echo "  ./scripts/docker_push.sh"
echo ""
