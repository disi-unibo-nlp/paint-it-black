#!/bin/bash
# Like quick_eval_bsub_qwen.sh but runs vllm with tensor parallelism across
# multiple GPUs — useful for larger Qwen VL models that don't fit on one A100.
#
# Usage:
#   ./experiments/quick_eval_bsub_qwen_parallel.sh
#   ./experiments/quick_eval_bsub_qwen_parallel.sh --n 10            # debug run
#   ./experiments/quick_eval_bsub_qwen_parallel.sh --split medium
#   ./experiments/quick_eval_bsub_qwen_parallel.sh --tensor_parallel 2  # override TP

DATASET_DIR=disi-unibo-nlp/paint-it-black
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
SPLIT=hard          # base, medium, or hard
N=""                # empty = full dataset
REASONING_PARSER=qwen3
KV_CACHE_FP8=0  # FP8 models benefit from fp8 kv cache
CONFIG=config/inference/base.yaml
FG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)     N="$2";     shift 2 ;;
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
./scripts/run_vllm_inference_bsub_parallel.sh \
    --model "$MODEL" \
    --vllm_max_model_len 24576 \
    --reasoning_parser "$REASONING_PARSER" \
    $([ "$KV_CACHE_FP8" -eq 1 ] && echo "--kv_cache_dtype fp8") \
    --mm_encoder_tp_mode data \
    --mm_processor_cache_type shm \
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
