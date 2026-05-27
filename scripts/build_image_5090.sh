#!/bin/bash
# Build the project Docker image for RTX 5090 (Blackwell, CUDA 12.9).
#
# Usage: ./scripts/build_image_5090.sh

IMAGE_NAME=deid

docker build -t $IMAGE_NAME -f docker/5090/Dockerfile .
