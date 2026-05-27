#!/bin/bash

ANNOTATION_DIR=data/chosen_reports
OUTPUT_DIR=./data/paint_it_black

./scripts/run_cont.sh python3 src/dataprep/build_dataset.py \
    --input_dir $ANNOTATION_DIR \
    --output $OUTPUT_DIR \
    --split base \
    --dpi 300