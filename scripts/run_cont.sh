#!/bin/bash
# Run a command inside the project Docker container, or start an interactive shell.
#
# Usage:
#   ./scripts/run_cont.sh                                                    # interactive shell
#   ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config ...  # run a command
#   ./scripts/run_cont.sh -j                                                 # start Jupyter on port 8888
#   ./scripts/run_cont.sh -j 8899                                            # start Jupyter on external port 8899

IMAGE_NAME=deid
CONT_WORKDIR=/workdir
JUPYTER_PORT=8888

# Load .env if present (set default if not specified)
set -a; [ -f .env ] && source .env; set +a

# Parse optional -j flag (must come before any command)
# Starts a Jupyter notebook server; maps EXTERNAL_PORT:8888 (default external = 8888)
PORT_ARGS=""
if [ "${1}" = "-j" ]; then
    shift
    if [[ "${1}" =~ ^[0-9]+$ ]]; then
        PORT_ARGS="-p ${1}:${JUPYTER_PORT}"
        shift
    else
        PORT_ARGS="-p ${JUPYTER_PORT}:${JUPYTER_PORT}"
    fi
    set -- jupyter notebook --ip=0.0.0.0 --no-browser --port=${JUPYTER_PORT} \
        --NotebookApp.token='' --NotebookApp.password='' --allow-root
fi

if [ $# -eq 0 ]; then
    docker run -it --rm \
        $PORT_ARGS \
        -v ./src:$CONT_WORKDIR/src \
        -v ./config:$CONT_WORKDIR/config \
        -v ./experiments:$CONT_WORKDIR/experiments \
        -v ./templates:$CONT_WORKDIR/templates \
        -v ./output:$CONT_WORKDIR/output \
        -v ./data:$CONT_WORKDIR/data \
        -m 30g \
        $IMAGE_NAME
else
    docker run --rm \
        $PORT_ARGS \
        -v ./src:$CONT_WORKDIR/src \
        -v ./config:$CONT_WORKDIR/config \
        -v ./experiments:$CONT_WORKDIR/experiments \
        -v ./templates:$CONT_WORKDIR/templates \
        -v ./output:$CONT_WORKDIR/output \
        -v ./data:$CONT_WORKDIR/data \
        -m 30g \
        $IMAGE_NAME "$@"
fi
