#!/bin/bash
# Push Parallel Synth Docker Images to Registry

set -e

# Configuration
DOCKER_USERNAME=${DOCKER_USERNAME:-"parallelsynth"}
IMAGE_NAME="dataset-generator"
VERSION=${VERSION:-"latest"}
REGISTRY=${REGISTRY:-"docker.io"}  # or ghcr.io for GitHub Container Registry

echo "=================================================="
echo "  Pushing Parallel Synth Docker Image"
echo "=================================================="
echo ""

# Check if logged in to Docker registry
echo "Checking Docker login..."
if ! docker info | grep -q "Username"; then
    echo "Not logged in to Docker registry."
    echo "Logging in..."
    docker login ${REGISTRY}
fi

echo ""
echo "Pushing images..."
echo ""

# Push version tag
echo "Pushing ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}

# Push latest tag
echo "Pushing ${DOCKER_USERNAME}/${IMAGE_NAME}:latest..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo ""
echo "=================================================="
echo "  Images pushed successfully!"
echo "=================================================="
echo ""
echo "Images available at:"
echo "  - ${REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo "  - ${REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "Pull with:"
echo "  docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "Use in CI/CD:"
echo "  docker run --rm --gpus all ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
