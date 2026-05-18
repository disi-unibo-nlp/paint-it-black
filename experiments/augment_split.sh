#!/bin/bash

INPUT_DATASET=./data/paint_it_black
INPUT_SPLIT=base
OUTPUT_SPLIT=medium
CONFIG=config/dataprep/low_noise_mix.yaml
SEED=1

./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --input_dataset $INPUT_DATASET \
    --input_split $INPUT_SPLIT \
    --output_split $OUTPUT_SPLIT \
    --config $CONFIG \
    --seed $SEED \
    --output_scale 0.48  # downscale from 300 to 144 DPI

OUTPUT_SPLIT=hard
CONFIG=config/dataprep/high_noise_mix.yaml
SEED=1

./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --input_dataset $INPUT_DATASET \
    --input_split $INPUT_SPLIT \
    --output_split $OUTPUT_SPLIT \
    --config $CONFIG \
    --seed $SEED \
    --output_scale 0.48  # downscale from 300 to 144 DPI