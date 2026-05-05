#!/bin/bash
# Run a command inside the project Docker container, or start an interactive shell.
# Uses --network host so the container can reach host services (e.g. vllm serve on :8000).
#
# Usage:
#   ./scripts/run_cont.sh                                                    # interactive shell
#   ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config ...  # run a command
#   ./scripts/run_cont.sh -j                                                 # start Jupyter on port 8888
#   ./scripts/run_cont.sh -j 8899                                            # start Jupyter on port 8899

IMAGE_NAME=deid
CONT_WORKDIR=/workdir
JUPYTER_PORT=8888
CUDA_VISIBLE_DEVICES=0

# Load .env if present (set default if not specified)
set -a; [ -f .env ] && source .env; set +a

# Parse optional -j flag (must come before any command)
# With --network host the port is exposed directly on the host; no -p mapping needed.
if [ "${1}" = "-j" ]; then
    shift
    if [[ "${1}" =~ ^[0-9]+$ ]]; then
        JUPYTER_PORT="${1}"
        shift
    fi
    set -- jupyter notebook --ip=0.0.0.0 --no-browser --port=${JUPYTER_PORT} \
        --NotebookApp.token='' --NotebookApp.password='' --allow-root
fi

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
