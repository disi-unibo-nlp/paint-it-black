#!/bin/bash
# Submit vllm serve via bsub (LSF, A100 GPU), then run inference locally.
# No Docker — inference runs in the current Python environment.
#
# Usage:
#   ./scripts/run_vllm_inference_bsub.sh \
#       --model Qwen/Qwen3.5-9B \
#       [--vllm_port 8000] \
#       [--vllm_gpu_memory_utilization 0.90] \
#       [--vllm_max_model_len 8192] \
#       [--vllm_tensor_parallel_size 1] \
#       [--bsub_mem 128G] \
#       [--bsub_gpu_model NVIDIAA100_SXM4_40GB] \
#       [--reasoning_parser qwen3] \
#       [--chat_template templates/gemma4_chat_template.jinja] \
#       [--enable_prefix_caching] \
#       [--kv_cache_dtype fp8] \
#       [--vision_tokens 1120] \
#       [--config config/inference/base.yaml] \
#       [--input_dataset ...] \
#       [any other run_inference.py arg ...]

MODEL=""
VLLM_PORT=8000
GPU_MEM=0.95
MAX_MODEL_LEN=8192
TENSOR_PARALLEL=1
BSUB_MEM=128G
BSUB_GPU_MODEL=NVIDIAH10080GBHBM3 #NVIDIAA100_SXM4_80GB #NVIDIAH10080GBHBM3 #NVIDIAA100_SXM4_80GB #NVIDIAH10080GBHBM3 #
REASONING_PARSER=""
CHAT_TEMPLATE=""
ENABLE_PREFIX_CACHING=0
KV_CACHE_DTYPE=""
VISION_TOKENS=""  # Gemma 4 only: 70|140|280|560|1120 tokens per image (default: server default)
TRUST_REMOTE_CODE=0
DTYPE=""
VIDEO_PRUNING_RATE=""
MM_ENCODER_TP_MODE=""
MM_PROCESSOR_CACHE_TYPE=""

INFERENCE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                        MODEL="$2";              shift 2 ;;
        --vllm_port)                    VLLM_PORT="$2";          shift 2 ;;
        --vllm_gpu_memory_utilization)  GPU_MEM="$2";            shift 2 ;;
        --vllm_max_model_len)           MAX_MODEL_LEN="$2";      shift 2 ;;
        --vllm_tensor_parallel_size)    TENSOR_PARALLEL="$2";    shift 2 ;;
        --bsub_mem)                     BSUB_MEM="$2";           shift 2 ;;
        --bsub_gpu_model)               BSUB_GPU_MODEL="$2";     shift 2 ;;
        --reasoning_parser)             REASONING_PARSER="$2";   shift 2 ;;
        --chat_template)                CHAT_TEMPLATE="$2";      shift 2 ;;
        --enable_prefix_caching)        ENABLE_PREFIX_CACHING=1; shift ;;
        --kv_cache_dtype)               KV_CACHE_DTYPE="$2";     shift 2 ;;
        --vision_tokens)                VISION_TOKENS="$2";      shift 2 ;;
        --trust_remote_code)            TRUST_REMOTE_CODE=1;     shift ;;
        --dtype)                        DTYPE="$2";                      shift 2 ;;
        --video_pruning_rate)           VIDEO_PRUNING_RATE="$2";         shift 2 ;;
        --mm_encoder_tp_mode)           MM_ENCODER_TP_MODE="$2";         shift 2 ;;
        --mm_processor_cache_type)      MM_PROCESSOR_CACHE_TYPE="$2";    shift 2 ;;
        *)  INFERENCE_ARGS+=("$1");     shift ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    exit 1
fi

# Load .env if present
set -a; [ -f .env ] && source .env; set +a

if [ -z "$HF_HUB_CACHE" ]; then
    echo "ERROR: HF_HUB_CACHE is not set — refusing to run to avoid re-downloading models."
    echo "Set HF_HUB_CACHE in your .env file (e.g. HF_HUB_CACHE=/dccstor/jcc/.cache/huggingface/hub)"
    exit 1
fi

mkdir -p output

MODEL_SAFE="${MODEL//\//_}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Absolute path so bsub resolves it correctly regardless of worker's $HOME
VLLM_LOG="$(pwd)/output/vllm_${MODEL_SAFE}_${TIMESTAMP}_$$.log"

echo "Submitting vllm serve job via bsub..."
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  vllm log: $VLLM_LOG"

# Write the vllm serve command to a temp script in the shared output dir.
# This avoids quoting hell when embedding JSON values (e.g. --mm-processor-kwargs)
# inside a bash -c "..." string passed through bsub.
VLLM_SCRIPT="$(pwd)/output/vllm_cmd_${TIMESTAMP}_$$.sh"
{
    echo '#!/bin/bash'
    echo 'set -e'
    printf 'stdbuf -oL -eL vllm serve %q' "$MODEL"
    printf ' --host 0.0.0.0'
    printf ' --port %s' "$VLLM_PORT"
    printf ' --gpu-memory-utilization %s' "$GPU_MEM"
    printf ' --max-model-len %s' "$MAX_MODEL_LEN"
    printf ' --tensor-parallel-size %s' "$TENSOR_PARALLEL"
    [ -n "$REASONING_PARSER" ]         && printf ' --reasoning-parser %s' "$REASONING_PARSER"
    [ -n "$CHAT_TEMPLATE" ]            && printf ' --chat-template %s' "$(realpath "$CHAT_TEMPLATE")"
    [ "$ENABLE_PREFIX_CACHING" -eq 1 ] && printf ' --enable-prefix-caching'
    [ -n "$KV_CACHE_DTYPE" ]           && printf ' --kv-cache-dtype %s' "$KV_CACHE_DTYPE"
    [ -n "$VISION_TOKENS" ]            && printf " --mm-processor-kwargs '{\"max_soft_tokens\": %s}'" "$VISION_TOKENS"
    [ "$TRUST_REMOTE_CODE" -eq 1 ]     && printf ' --trust-remote-code'
    [ -n "$DTYPE" ]                    && printf ' --dtype %s' "$DTYPE"
    [ -n "$VIDEO_PRUNING_RATE" ]       && printf ' --video-pruning-rate %s' "$VIDEO_PRUNING_RATE"
    [ -n "$MM_ENCODER_TP_MODE" ]       && printf ' --mm-encoder-tp-mode %s' "$MM_ENCODER_TP_MODE"
    [ -n "$MM_PROCESSOR_CACHE_TYPE" ]  && printf ' --mm-processor-cache-type %s' "$MM_PROCESSOR_CACHE_TYPE"
    printf ' > %q 2>&1\n' "$VLLM_LOG"
} > "$VLLM_SCRIPT"
chmod +x "$VLLM_SCRIPT"
echo "  vllm script: $VLLM_SCRIPT"

# Find hosts that already have a vllm server on VLLM_PORT and exclude them.
BUSY_HOSTS=$(bjobs -noheader -o "exec_host" 2>/dev/null \
    | sed 's/:.*//' | sed 's/^[0-9]*\*//' | sort -u)
R_EXPR=""
for h in $BUSY_HOSTS; do
    if curl -sf --max-time 2 "http://${h}:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "  host $h is busy on port $VLLM_PORT — excluding from bsub"
        R_EXPR="${R_EXPR:+$R_EXPR && }hname!='$h'"
    fi
done

# PYTHONUNBUFFERED=1 disables Python's internal buffer so lines appear immediately.
# stdbuf -oL -eL forces line-buffered mode on the redirected streams.
# Together these make `tail -f $VLLM_LOG` show output live.
BSUB_OUT=$(bsub -M "$BSUB_MEM" -n 1 -o /dev/null \
    ${R_EXPR:+-R "$R_EXPR"} \
    -gpu "num=${TENSOR_PARALLEL}:mode=exclusive_process:gmodel=${BSUB_GPU_MODEL}" \
    -env "HF_HUB_CACHE=$HF_HUB_CACHE,HF_TOKEN=$HF_TOKEN,PYTHONUNBUFFERED=1,all" \
    bash "$VLLM_SCRIPT" \
    2>&1)

echo "$BSUB_OUT"

# Extract job ID from: Job <12345> is submitted to ...
JOB_ID=$(echo "$BSUB_OUT" | grep -oP '(?<=Job <)\d+(?=>)')
if [ -z "$JOB_ID" ]; then
    echo "ERROR: could not parse bsub job ID from output"
    exit 1
fi
echo "LSF job ID: $JOB_ID"

# ── Wait until job transitions from PEND to RUN ──────────────────────────────
echo "Waiting for job $JOB_ID to start running..."
while true; do
    STATUS=$(bjobs -noheader "$JOB_ID" 2>/dev/null | awk '{print $3}')
    if [ "$STATUS" = "RUN" ]; then
        break
    elif [ "$STATUS" = "EXIT" ] || [ "$STATUS" = "DONE" ]; then
        echo "ERROR: job $JOB_ID ended unexpectedly (status: $STATUS)"
        echo "Check log: $VLLM_LOG"
        exit 1
    fi
    echo "  job status: ${STATUS:-UNKNOWN} — waiting 10s..."
    sleep 10
done

# ── Get execution host ────────────────────────────────────────────────────────
EXEC_HOST=$(bjobs -noheader -o "exec_host" "$JOB_ID" 2>/dev/null | sed 's/:.*//' | sed 's/^[0-9]*\*//')
if [ -z "$EXEC_HOST" ]; then
    echo "ERROR: could not determine exec host for job $JOB_ID"
    bkill "$JOB_ID" 2>/dev/null
    exit 1
fi
echo "vllm running on host: $EXEC_HOST"

VLLM_URL="http://${EXEC_HOST}:${VLLM_PORT}"

# ── Wait for vllm health endpoint ────────────────────────────────────────────
echo "Waiting for vllm server at $VLLM_URL/health serving model '$MODEL'..."
while true; do
    if curl -sf "$VLLM_URL/health" > /dev/null 2>&1; then
        # Health is up — verify it's our model, not a stale server
        SERVED=$(curl -sf "$VLLM_URL/v1/models" 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        if [ "$SERVED" = "$MODEL" ]; then
            break
        fi
        echo "  health up but serving '$SERVED', waiting for '$MODEL'..."
    fi
    # Check job is still alive
    STATUS=$(bjobs -noheader "$JOB_ID" 2>/dev/null | awk '{print $3}')
    if [ "$STATUS" = "EXIT" ] || [ "$STATUS" = "DONE" ]; then
        echo "ERROR: vllm job died before server became ready"
        echo "Check log: $VLLM_LOG"
        exit 1
    fi
    sleep 10
    echo "  still waiting..."
done
echo "vllm server ready at $VLLM_URL"
echo "vllm log: $VLLM_LOG"

# ── Run inference ─────────────────────────────────────────────────────────────
python3 src/inference/run_inference.py \
    --model "$MODEL" \
    --base_url "$VLLM_URL/v1" \
    "${INFERENCE_ARGS[@]}"
INFERENCE_EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "Killing vllm job $JOB_ID..."
bkill "$JOB_ID" 2>/dev/null

exit $INFERENCE_EXIT
