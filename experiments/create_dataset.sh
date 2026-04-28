#!/bin/bash

# NOTE: these directories are not included in the repo, but should be created by the user and populated with the appropriate data before running this script. See README for details.
ANNOTATION_DIR=./data/dev_annotation_cases
OUTPUT_DIR=./data/test_ds

# test on a few annotated pdfs
python3 src/dataprep/build_dataset.py --input_dir $ANNOTATION_DIR --output_dir $OUTPUT_DIR --split base --dpi 200
python3 src/dataprep/augment_pdfs.py --input_dataset $OUTPUT_DIR --input_split base --output_split medium --config config/dataprep/low_noise_mix.yaml --seed 1
python3 src/dataprep/augment_pdfs.py --input_dataset $OUTPUT_DIR --input_split base --output_split hard --config config/dataprep/high_noise_mix.yaml --seed 1
python3 src/dataprep/augment_pdfs.py --input_dataset $OUTPUT_DIR --input_split base --output_split hard --config config/dataprep/high_noise_mix.yaml --seed 2 --resample_ids 0 5 7