#!/bin/bash
# Run a command inside the project Docker container, or start an interactive shell.
#
# Usage:
#   ./scripts/run_cont.sh                                                    # interactive shell
#   ./scripts/run_cont.sh python3 src/dataprep/augment_pdfs.py --config ...  # run a command

IMAGE_NAME=deid
CONT_WORKDIR=/workdir

# Load .env if present (set default if not specified)
set -a; [ -f .env ] && source .env; set +a

if [ $# -eq 0 ]; then
    docker run -it --rm \
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
        -v ./src:$CONT_WORKDIR/src \
        -v ./config:$CONT_WORKDIR/config \
        -v ./experiments:$CONT_WORKDIR/experiments \
        -v ./templates:$CONT_WORKDIR/templates \
        -v ./output:$CONT_WORKDIR/output \
        -v ./data:$CONT_WORKDIR/data \
        -m 30g \
        $IMAGE_NAME "$@"
fi
