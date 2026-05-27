#!/bin/bash
# Inference benchmark experiments.
# Called directly from the host (NOT via run_cont.sh).
# run_vllm_inference.sh starts VLLM serve and the inference script in the same container.
#
# Usage:
#   ./experiments/run_inference.sh

DATASET_DIR=disi-unibo-nlp/paint-it-black
MODELS=(
    Qwen/Qwen3.5-9B
    # ...
    )
SPLIT=base
RUN_NAME=e1_q35-9B_base

# Run evaluations for base, medium, and hard splits

for MODEL in "${MODELS[@]}"; do
    for SPLIT in base medium hard; do
        RUN_NAME=e1_q35-9B_${SPLIT}
        CONFIG=config/inference/base.yaml
        echo "Running inference for split: $SPLIT"
        ./scripts/run_vllm_inference.sh \
            --model $MODEL \
            --vllm_max_model_len 24576 \
            --config $CONFIG \
            --input_dataset $DATASET_DIR \
            --input_split $SPLIT \
            --from_hub \
            --run_name $RUN_NAME
    done
done

