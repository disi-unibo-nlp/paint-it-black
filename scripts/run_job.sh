#!/bin/bash
# Submit a command as a SLURM job running inside the Docker container.
#
# Usage: ./scripts/run_job.sh <command>
# Example: ./scripts/run_job.sh python3 src/dataprep/augment_pdfs.py --config dataprep/augment_config

COMMAND="$@"

sbatch \
    --gpus=1 \
    ./scripts/run_cont.sh $COMMAND

# Common sbatch options to add as needed:
#   -w <node_name>          pin to a specific node
#   --time=24:00:00         wall-clock time limit
#   --mem=32G               RAM limit
#   --job-name=my_run       job name in SLURM queue
