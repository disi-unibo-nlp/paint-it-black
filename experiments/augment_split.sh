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
    --seed $SEED

OUTPUT_SPLIT=hard
CONFIG=config/dataprep/high_noise_mix.yaml
SEED=1

./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --input_dataset $INPUT_DATASET \
    --input_split $INPUT_SPLIT \
    --output_split $OUTPUT_SPLIT \
    --config $CONFIG \
    --seed $SEED

# resample hard
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
    --resample_ids 5 13 38 42 46 47 56 58 65 77 78 100 109 116 120 125 135 136