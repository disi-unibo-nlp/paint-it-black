#!/bin/bash
# Build the project Docker image for RTX 3090 (Ampere, CUDA 12.4).
#
# Usage: ./scripts/build_image_3090.sh

IMAGE_NAME=deid

docker build -t $IMAGE_NAME -f docker/3090/Dockerfile .
