#!/bin/bash
# Inference benchmark experiments.
# Called directly from the host (NOT via run_cont.sh).
# run_vllm_inference.sh starts VLLM serve and the inference script in the same container.
#
# Usage:
#   ./experiments/run_inference.sh

DATASET_DIR=dfreddi/multimodal-deid
MODEL=Qwen/Qwen2.5-VL-3B-Instruct

# ── 1. Base split (clean, no augmentation) ───────────────────────────────────
./scripts/run_vllm_inference.sh \
    --model $MODEL \
    --config config/inference/base.yaml \
    --input_dataset $DATASET_DIR \
    --input_split base \
    --from_hub \
    --run_name eval_base

# ── 2. Medium split (low-noise augmentation) ─────────────────────────────────
./scripts/run_vllm_inference.sh \
    --model $MODEL \
    --config config/inference/base.yaml \
    --input_dataset $DATASET_DIR \
    --input_split medium \
    --from_hub \
    --run_name eval_medium

# ── 3. Hard split (high-noise augmentation) ──────────────────────────────────
./scripts/run_vllm_inference.sh \
    --model $MODEL \
    --config config/inference/base.yaml \
    --input_dataset $DATASET_DIR \
    --input_split hard \
    --from_hub \
    --run_name eval_hard

# ── 4. Base split with stricter IoU thresholds ───────────────────────────────
# ./scripts/run_vllm_inference.sh \
#     --model $MODEL \
#     --config config/inference/base.yaml \
#     --input_dataset $DATASET_DIR \
#     --input_split base \
#     --from_hub \
#     --iou_thresholds 0.5 0.75 0.9 \
#     --run_name eval_base_strict_iou

# ── 5. Remote OpenAI backend (GPT-4o) ────────────────────────────────────────
# Uncomment and set OPENAI_API_KEY to run (no VLLM needed).
# ./scripts/run_cont.sh python3 src/inference/run_inference.py \
#     --config config/inference/openai.yaml \
#     --input_dataset $DATASET_DIR \
#     --input_split base \
#     --from_hub \
#     --api_key $OPENAI_API_KEY \
#     --run_name eval_gpt4o_base
