#!/bin/bash
# Run all GLiNER NER baselines on the base split.
# Runs inside the project Docker container via run_cont.sh.
#
# Usage:
#   ./experiments/baselines/run_gliner_baselines.sh
#   ./experiments/baselines/run_gliner_baselines.sh --split medium
#   ./experiments/baselines/run_gliner_baselines.sh --max_samples 10

DATASET=disi-unibo-nlp/paint-it-black
SPLIT=base

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)       SPLIT="$2";       shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

CONFIGS=(
    gliner_pii
    gliner_multi_pii
    gliner_pii_large
    gliner2_privacy
    openbioner
)

for CONFIG in "${CONFIGS[@]}"; do
    RUN_NAME="${CONFIG}_${SPLIT}"
    echo ""
    echo "=========================================="
    echo "Running: $CONFIG  (split: $SPLIT)"
    echo "=========================================="

    CMD="python3 src/inference/run_ner_baseline.py \
        --config $CONFIG \
        --input_dataset $DATASET \
        --input_split $SPLIT \
        --from_hub \
        --output_dir output/ner_baseline \
        --run_name $RUN_NAME"

    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi

    ./scripts/run_cont.sh $CMD
done

echo ""
echo "All NER baselines complete. Results in output/ner_baseline/"
