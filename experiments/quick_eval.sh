#!/bin/bash
# Quick sanity-check: run inference on a small subset, then render predictions.
# Useful for iterating on prompts without a full benchmark run.
# Called directly from the host (NOT via run_cont.sh).
#
# Usage:
#   ./experiments/quick_eval.sh
#   ./experiments/quick_eval.sh --n 10        # number of samples (default 5)
#   ./experiments/quick_eval.sh --split medium # dataset split (default base)

DATASET_DIR=dfreddi/multimodal-deid
MODEL=Qwen/Qwen3.5-9B
SPLIT=base
N=10
RUN_NAME=q35-9B_quick_eval

# ── 1. Run inference ──────────────────────────────────────────────────────────
./scripts/run_vllm_inference.sh \
    --model $MODEL \
    --vllm_max_model_len 24576 \
    --config config/inference/base.yaml \
    --input_dataset $DATASET_DIR \
    --input_split $SPLIT \
    --from_hub \
    --max_samples $N \
    --run_name $RUN_NAME

# ── 2. Find the output dir just created ───────────────────────────────────────
RESULTS_JSON=output/inference/${RUN_NAME}/results.json

if [ ! -f "$RESULTS_JSON" ]; then
    echo "ERROR: could not find $RESULTS_JSON"
    exit 1
fi

echo ""
echo "Rendering $N samples from: $RESULTS_JSON"

# ── 3. Render predictions ──────────────────────────────────────────────────────
./scripts/run_cont.sh python3 src/inference/render_predictions.py \
    --results "$RESULTS_JSON" \
    --limit $N

echo ""
echo "Done. Check renders in: $(dirname $RESULTS_JSON)/renders/"
