#!/bin/bash
# Push a 300 DPI test dataset to HuggingFace Hub (10 instances per split).
# Uses the full-resolution local dataset; no downscaling is applied.
# Increase --max_instances or remove it once the repo quota allows more data.

INPUT_DATASET=./data/paint_it_black
HUB_DATASET=disi-unibo-nlp/paint-it-black-300dpi

./scripts/run_cont.sh python3 src/dataprep/push_to_hub.py \
    --input_dataset $INPUT_DATASET \
    --hub_dataset   $HUB_DATASET \
    --splits base medium hard \
    --target_dpi 300 \
    --max_instances 10 \
    --private
