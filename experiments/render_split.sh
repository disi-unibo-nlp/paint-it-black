#!/bin/bash

DATASET_NAME=dfreddi/multimodal-deid
SPLIT=base

python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT

SPLIT=medium

python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT