#!/bin/bash

RUN_NAME=output/inference/e1_q35-9B_base/results.json
N=20

./scripts/run_cont.sh python3 src/inference/render_predictions.py \
    --results "$RUN_NAME" \
    --limit $N