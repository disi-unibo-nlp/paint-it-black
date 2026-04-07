#!/bin/bash
# Build the project Docker image.
#
# Usage: ./scripts/build_image.sh

IMAGE_NAME=deid

docker build -t $IMAGE_NAME -f docker/Dockerfile .
