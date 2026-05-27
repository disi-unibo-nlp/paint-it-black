#!/bin/bash
# Resample specific rows of the hard split.
# Indices correspond to render file names (0-based).
# Re-augments only the listed rows; all other rows are kept unchanged.

INPUT_DATASET=./data/paint_it_black
INPUT_SPLIT=base
OUTPUT_SPLIT=hard
CONFIG=config/dataprep/high_noise_mix.yaml
SEED=2

./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --input_dataset $INPUT_DATASET \
    --input_split $INPUT_SPLIT \
    --output_dataset $INPUT_DATASET \
    --output_split $OUTPUT_SPLIT \
    --config $CONFIG \
    --seed $SEED \
    --output_scale 0.48 \
    --resample_ids 5 13 38 42 46 47 56 58 65 77 78 100 109 116 120 125 135 136
