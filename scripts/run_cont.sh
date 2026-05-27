#!/bin/bash
# Run a command inside the project Docker container, or start an interactive shell.
# Uses --network host so the container can reach host services (e.g. vllm serve on :8000).
#
# Usage:
#   ./scripts/run_cont.sh                                                    # interactive shell
#   ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config ...  # run a command

IMAGE_NAME=deid
CONT_WORKDIR=/workdir
CUDA_VISIBLE_DEVICES=0 # comment this if using run_cont in slurm

# Load .env if present (set default if not specified)
set -a; [ -f .env ] && source .env; set +a

if [ $# -eq 0 ]; then
    docker run -it --rm \
        --network host \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        -v ./src:$CONT_WORKDIR/src \
        -v ./config:$CONT_WORKDIR/config \
        -v ./experiments:$CONT_WORKDIR/experiments \
        -v ./templates:$CONT_WORKDIR/templates \
        -v ./output:$CONT_WORKDIR/output \
        -v ./data:$CONT_WORKDIR/data \
        -m 30g \
        -e HF_TOKEN=$HF_TOKEN \
        -e HF_HOME=$HF_HOME \
        -v $HF_HOME:$CONT_WORKDIR/$HF_HOME \
        $IMAGE_NAME
else
    docker run --rm \
        --network host \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        -v ./src:$CONT_WORKDIR/src \
        -v ./config:$CONT_WORKDIR/config \
        -v ./experiments:$CONT_WORKDIR/experiments \
        -v ./templates:$CONT_WORKDIR/templates \
        -v ./output:$CONT_WORKDIR/output \
        -v ./data:$CONT_WORKDIR/data \
        -m 30g \
        -e HF_TOKEN=$HF_TOKEN \
        -e HF_HOME=$HF_HOME \
        -v $HF_HOME:$CONT_WORKDIR/$HF_HOME \
        $IMAGE_NAME "$@"
fi
