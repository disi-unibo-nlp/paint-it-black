#!/bin/bash
# Like quick_eval.sh but runs on the LSF cluster via bsub (no Docker).
# vllm serve is submitted as an LSF job on an A100 GPU node;
# inference runs locally once the server is up.
#
# Usage:
#   ./experiments/quick_eval_bsub.sh
#   ./experiments/quick_eval_bsub.sh --n 10        # number of samples (default 10)
#   ./experiments/quick_eval_bsub.sh --split medium # dataset split (default base)

DATASET_DIR=disi-unibo-nlp/paint-it-black-2 #disi-unibo-nlp/paint-it-black
MODEL=google/gemma-4-26B-A4B-it #google/gemma-4-31B-it #google/gemma-4-26B-A4B-it #Qwen/Qwen3.5-27B-FP8
SPLIT=medium        # base, medium, or hard
N=""               # empty = full dataset; pass --n 10 for a debug run
REASONING_PARSER=gemma4 #qwen3  # set to "" to disable
CHAT_TEMPLATE=templates/gemma4_chat_template.jinja  # set to "" for non-Gemma4 models
CONFIG=config/inference/gemma4.yaml  # gemma4.yaml disables guided_json (incompatible with reasoning parser)
KV_CACHE_FP8=0  # set to 1 to pass --kv_cache_dtype fp8
VISION_TOKENS=1120  # Gemma 4 vision token budget: 70|140|280|560|1120 (higher = more detail, more compute)
FG=0   # set by --fg when called from run_all_splits_bsub.sh

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)                N="$2";               shift 2 ;;
        --split)            SPLIT="$2";           shift 2 ;;
        --reasoning_parser) REASONING_PARSER="$2"; shift 2 ;;
        --kv_cache_fp8)     KV_CACHE_FP8=1;       shift ;;
        --vision_tokens)    VISION_TOKENS="$2";   shift 2 ;;
        --fg)               FG=1;                 shift ;;
        *)                  shift ;;
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
    ${REASONING_PARSER:+--reasoning_parser "$REASONING_PARSER"} \
    ${CHAT_TEMPLATE:+--chat_template "$CHAT_TEMPLATE"} \
    ${VISION_TOKENS:+--vision_tokens "$VISION_TOKENS"} \
    --enable_prefix_caching \
    $([ "$KV_CACHE_FP8" -eq 1 ] && echo "--kv_cache_dtype fp8") \
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
