#!/bin/bash

ANNOTATION_DIR=data/chosen_reports
OUTPUT_DIR=disi-unibo-nlp/paint-it-black

./scripts/run_cont.sh python3 src/dataprep/build_dataset.py \
    --input_dir $ANNOTATION_DIR \
    --output $OUTPUT_DIR \
    --push_to_hub \
    --split base \
    --dpi 300