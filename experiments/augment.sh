#!/bin/bash

for config in config/dataprep/augment_examples/*; do
    [ -f "$config" ] || continue
    python3 src/dataprep/augment_pdfs.py --num_augmentations 8 --config "$config" --flat
done

for config in config/dataprep/low_noise/*; do
    [ -f "$config" ] || continue
    python3 src/dataprep/augment_pdfs.py --num_augmentations 8 --config "$config" --flat
done

python3 src/dataprep/augment_pdfs.py --num_augmentations 10 --config config/dataprep/low_noise_mix.yaml --flat

python3 src/dataprep/augment_pdfs.py \
    --num_augmentations 1 \
    --config config/dataprep/low_noise_mix.yaml \
    --input_dir ./data/dev_docs \
    --output_dir ./output/dev_docs_low_noise