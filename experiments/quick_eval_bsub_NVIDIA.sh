#!/bin/bash
# Pre-configured for NVIDIA Nemotron VL models.
# vllm serve is submitted as an LSF job on an A100 GPU node;
# inference runs locally once the server is up.
#
# Usage:
#   ./experiments/quick_eval_bsub_NVIDIA.sh
#   ./experiments/quick_eval_bsub_NVIDIA.sh --n 10         # number of samples (default: full)
#   ./experiments/quick_eval_bsub_NVIDIA.sh --split medium # dataset split (default: base)

DATASET_DIR=disi-unibo-nlp/paint-it-black
MODEL=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
SPLIT=base # base, medium, or hard
N=""       # empty = full dataset; pass --n 10 for a debug run
CONFIG=config/inference/base.yaml
FG=0       # set by --fg when called from run_all_splits_bsub.sh

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)     N="$2";    shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --fg)    FG=1;       shift ;;
        *)       shift ;;
    esac
done

MODEL_SHORT=${MODEL##*/}
RUN_NAME=${MODEL_SHORT}_${SPLIT}_${N:-all}_$(date +%Y%m%d_%H%M%S)

# ── Self-background unless --fg was passed ────────────────────────────────────
if [ "$FG" -eq 0 ]; then
    mkdir -p output
    LOG="output/run_${RUN_NAME}.log"
    nohup "$0" "$@" --fg > "$LOG" 2>&1 &
    echo "Job submitted. Monitor with:"
    echo "  tail -f $LOG"
    exit 0
fi

# ── 1. Run inference (via LSF bsub) ──────────────────────────────────────────
./scripts/run_vllm_inference_bsub.sh \
    --model "$MODEL" \
    --vllm_max_model_len 24576 \
    --dtype bfloat16 \
    --video_pruning_rate 0 \
    --trust_remote_code \
    --enable_prefix_caching \
    --config "$CONFIG" \
    --input_dataset "$DATASET_DIR" \
    --input_split "$SPLIT" \
    --from_hub \
    ${N:+--max_samples "$N"} \
    --run_name "$RUN_NAME"

# ── 2. Find the output dir just created ───────────────────────────────────────
RESULTS_JSON=output/inference/${RUN_NAME}/results.json

if [ ! -f "$RESULTS_JSON" ]; then
    echo "ERROR: could not find $RESULTS_JSON"
    exit 1
fi

echo ""
echo "Rendering samples from: $RESULTS_JSON"

# ── 3. Render predictions ──────────────────────────────────────────────────────
python3 src/inference/render_predictions.py \
    --results "$RESULTS_JSON" \
    ${N:+--limit "$N"}

echo ""
echo "Done. Check renders in: $(dirname $RESULTS_JSON)/renders/"
