#!/bin/bash
# Run the PDF augmentation pipeline.
#
# Usage:
#   ./scripts/run_cont.sh bash experiments/augment.sh
#   ./scripts/run_cont.sh bash experiments/augment.sh --num_augmentations 3

for config in config/dataprep/augment_examples/*; do
    [ -f "$config" ] || continue
    python3 src/dataprep/augment_pdfs.py --num_augmentations 8 --config "$config" --flat "$@"
done

for config in config/dataprep/low_noise/*; do
    [ -f "$config" ] || continue
    python3 src/dataprep/augment_pdfs.py --num_augmentations 8 --config "$config" --flat "$@"
done