#!/bin/bash

DATASET_NAME=disi-unibo-nlp/paint-it-black

SPLIT=base
./scripts/run_cont.sh python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT

SPLIT=medium
./scripts/run_cont.sh python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT

SPLIT=hard
./scripts/run_cont.sh python3 src/analysis/render_dataset.py \
    --dataset $DATASET_NAME \
    --split $SPLIT