#!/bin/bash

# all library augmentations showcase
# for config in config/dataprep/augment_examples/*; do
#     [ -f "$config" ] || continue
#     ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config "$config" --flat
# done

# low noise augmentations for tuning
for config in config/dataprep/low_noise/*; do
    [ -f "$config" ] || continue
    ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config "$config" --flat
done

# actual sampling for dev set with low noise config
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config config/dataprep/low_noise_mix.yaml --flat

./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --num_augmentations 1 \
    --config config/dataprep/low_noise_mix.yaml \
    --input_dir ./data/dev_docs \
    --output_dir ./output/dev_docs_low_noise

# showcase of all augmentations on high noise config
for config in config/dataprep/high_noise/*; do
    [ -f "$config" ] || continue
    ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config "$config" --flat
done

# actual sampling for dev set with high noise config
./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config config/dataprep/high_noise_mix.yaml --flat

./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py \
    --num_augmentations 1 \
    --config config/dataprep/high_noise_mix.yaml \
    --input_dir ./data/dev_docs \
    --output_dir ./output/dev_docs_high_noise