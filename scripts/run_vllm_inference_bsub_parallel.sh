#!/bin/bash
# Wrapper around run_vllm_inference_bsub.sh that requests 4 GPUs by default
# for tensor-parallel inference of large models.
#
# All arguments are forwarded to run_vllm_inference_bsub.sh.
# Defaults (--vllm_tensor_parallel_size 4, --bsub_mem 256G) can be overridden
# by passing the same flags explicitly — they are processed last so they win.
#
# Usage: same as run_vllm_inference_bsub.sh

exec "$(dirname "$0")/run_vllm_inference_bsub.sh" \
    --vllm_tensor_parallel_size 4 \
    --bsub_mem 256G \
    "$@"
