#!/bin/bash
# Run VLLM serve + inference in a single container.
# Both processes share the same network namespace, so localhost:PORT always works.
#
# Usage (called directly from the host, NOT via run_cont.sh):
#   ./scripts/run_vllm_inference.sh \
#       --model Qwen/Qwen3-VL-2B-Thinking \
#       [--vllm_port 8000] \
#       [--vllm_gpu_memory_utilization 0.9] \
#       [--vllm_max_model_len 4096] \
#       [--vllm_tensor_parallel_size 1] \
#       [--config config/inference/base.yaml] \
#       [--input_dataset data/test_ds] \
#       [--input_split base] \
#       [--run_name my_run] \
#       [any other run_inference.py arg ...]
#
# --model and --vllm_* are consumed here; everything else is forwarded to run_inference.py.

IMAGE_NAME=deid
CONT_WORKDIR=/workdir
CUDA_VISIBLE_DEVICES=0

# VLLM serve defaults
MODEL=""
VLLM_PORT=8000
GPU_MEM=0.9
MAX_MODEL_LEN=8192
TENSOR_PARALLEL=1

# Collect inference args separately
INFERENCE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                        MODEL="$2";           shift 2 ;;
        --vllm_port)                    VLLM_PORT="$2";       shift 2 ;;
        --vllm_gpu_memory_utilization)  GPU_MEM="$2";         shift 2 ;;
        --vllm_max_model_len)           MAX_MODEL_LEN="$2";   shift 2 ;;
        --vllm_tensor_parallel_size)    TENSOR_PARALLEL="$2"; shift 2 ;;
        *)  INFERENCE_ARGS+=("$1");     shift ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    exit 1
fi

# Load .env if present
set -a; [ -f .env ] && source .env; set +a

# Build the quoted inference args string for bash -c
INFERENCE_ARGS_STR=""
for arg in "${INFERENCE_ARGS[@]}"; do
    INFERENCE_ARGS_STR="$INFERENCE_ARGS_STR $(printf '%q' "$arg")"
done

INNER_CMD="
vllm serve '$MODEL' \
    --port $VLLM_PORT \
    --gpu-memory-utilization $GPU_MEM \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL &
VLLM_PID=\$!

echo '[run_vllm_inference] Waiting for VLLM server on port $VLLM_PORT ...'
until curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; do
    sleep 5
    echo '[run_vllm_inference] Still waiting...'
done
echo '[run_vllm_inference] VLLM server ready.'

python3 src/inference/run_inference.py \
    --model '$MODEL' \
    --base_url http://localhost:$VLLM_PORT/v1 \
    $INFERENCE_ARGS_STR

kill \$VLLM_PID
wait \$VLLM_PID 2>/dev/null || true
"

docker run --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    -v ./src:$CONT_WORKDIR/src \
    -v ./config:$CONT_WORKDIR/config \
    -v ./experiments:$CONT_WORKDIR/experiments \
    -v ./templates:$CONT_WORKDIR/templates \
    -v ./output:$CONT_WORKDIR/output \
    -v ./data:$CONT_WORKDIR/data \
    -m 30g \
    $IMAGE_NAME bash -c "$INNER_CMD"
