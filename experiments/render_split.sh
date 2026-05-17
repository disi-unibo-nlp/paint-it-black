#!/bin/bash

DATASET_NAME=disi-unibo-nlp/paint-it-black

SPLIT=base
python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT

SPLIT=medium
python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT

SPLIT=hard
python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT