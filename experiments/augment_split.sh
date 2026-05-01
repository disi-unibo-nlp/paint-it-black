#!/bin/bash

INPUT_DATASET=dfreddi/multimodal-deid
INPUT_SPLIT=base
OUTPUT_SPLIT=medium
CONFIG=config/dataprep/low_noise_mix.yaml
SEED=1

# python3 src/dataprep/augment_pdfs.py \
#     --input_dataset $INPUT_DATASET \
#     --input_split $INPUT_SPLIT \
#     --output_split $OUTPUT_SPLIT \
#     --config $CONFIG \
#     --seed $SEED \
#     --push_to_hub

OUTPUT_SPLIT=hard
CONFIG=config/dataprep/high_noise_mix.yaml
SEED=1

python3 src/dataprep/augment_pdfs.py \
    --input_dataset $INPUT_DATASET \
    --input_split $INPUT_SPLIT \
    --output_split $OUTPUT_SPLIT \
    --config $CONFIG \
    --seed $SEED \
    --push_to_hub