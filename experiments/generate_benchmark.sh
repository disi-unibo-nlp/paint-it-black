#!/bin/bash

# Preview what will be generated (no API calls)
# ./scripts/run_cont.sh python3 src/dataprep/generate_benchmark.py \
#     --config dataprep/generate_benchmark --dry_run

# Generate 10 random MRI variants
./scripts/run_cont.sh python3 src/dataprep/generate_benchmark.py \
    --config dataprep/generate_benchmark \
    --types mri \
    --count 10

# Full grid across all types (omit --types and --count):
# ./scripts/run_cont.sh python3 src/dataprep/generate_benchmark.py \
#     --config dataprep/generate_benchmark
