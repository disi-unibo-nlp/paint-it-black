#!/bin/bash
# Run a command inside the project Docker container, or start an interactive shell.
#
# Usage:
#   ./scripts/run_cont.sh                                                    # interactive shell
#   ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config ...  # run a command
#   ./scripts/run_cont.sh -p 5151 python3 src/analysis/visualize_dataset.py  # expose a port (host:container)
#   ./scripts/run_cont.sh -p python3 src/analysis/visualize_dataset.py       # expose port 37341 (default)

IMAGE_NAME=deid
CONT_WORKDIR=/workdir
DEFAULT_EXT_PORT=37341
INTERNAL_PORT=5151

# Load .env if present (set default if not specified)
set -a; [ -f .env ] && source .env; set +a

# Parse optional -p flag (must come before the command)
# Maps EXTERNAL_PORT:INTERNAL_PORT — internal port is always $INTERNAL_PORT
PORT_ARGS=""
if [ "${1}" = "-p" ]; then
    shift
    # If next arg looks like a port number, use it as the external port; otherwise use the default
    if [[ "${1}" =~ ^[0-9]+$ ]]; then
        PORT_ARGS="-p ${1}:${INTERNAL_PORT}"
        shift
    else
        PORT_ARGS="-p ${DEFAULT_EXT_PORT}:${INTERNAL_PORT}"
    fi
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
