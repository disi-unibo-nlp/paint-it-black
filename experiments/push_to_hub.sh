#!/bin/bash
# Resize all splits to 144 DPI and push to HuggingFace Hub.
# base (300 DPI) is downscaled; medium and hard (already ~144 DPI) are uploaded as-is.

INPUT_DATASET=./data/paint_it_black
HUB_DATASET=disi-unibo-nlp/paint-it-black

./scripts/run_cont.sh python3 src/dataprep/push_to_hub.py \
    --input_dataset $INPUT_DATASET \
    --hub_dataset   $HUB_DATASET \
    --splits medium \
    --target_dpi 144 \
    --private
