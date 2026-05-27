#!/bin/bash
# Like quick_eval_bsub.sh but pre-configured for Nemotron models.
# vllm serve is submitted as an LSF job on an A100 GPU node;
# inference runs locally once the server is up.
#
# Usage:
#   ./experiments/quick_eval_bsub_qwen.sh
#   ./experiments/quick_eval_bsub_qwen.sh --n 10        # number of samples (default 10)
#   ./experiments/quick_eval_bsub_qwen.sh --split medium # dataset split (default base)

DATASET_DIR=disi-unibo-nlp/paint-it-black-2
MODEL=Qwen/Qwen3.6-35B-A3B #Qwen/Qwen3.5-27B-FP8 #qwen/Qwen3.5-9B #Qwen3.6-35B-A3B
SPLIT=medium # base, medium, or hard
N=""               # empty = full dataset; pass --n 10 for a debug run
REASONING_PARSER="qwen3"     
CHAT_TEMPLATE=""        # void — use vllm's built-in chat template
CONFIG=config/inference/base.yaml
KV_CACHE_FP8=0  # FP8 models benefit from fp8 kv cache
VISION_TOKENS="" # not used for Qwen — leave empty
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
    --trust_remote_code \
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
