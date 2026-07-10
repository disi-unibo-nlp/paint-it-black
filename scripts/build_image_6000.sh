#!/bin/bash
# Build the project Docker image for RTX 6000

IMAGE_NAME=deid

docker build -t $IMAGE_NAME -f docker/6000/Dockerfile .
