#!/bin/bash
# Run inference on all three splits in parallel.
# Each split gets its own bsub GPU job for vllm serve.
#
# Usage:
#   ./experiments/run_all_splits_bsub.sh
#   ./experiments/run_all_splits_bsub.sh --n 10                    # debug: 10 samples per split
#   ./experiments/run_all_splits_bsub.sh --reasoning_parser ""     # disable reasoning parser

REASONING_PARSER=qwen3  # set to "" to disable
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reasoning_parser) REASONING_PARSER="$2"; shift 2 ;;
        *)                  ARGS+=("$1");           shift ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)

mkdir -p output

PARSER_ARG=( ${REASONING_PARSER:+--reasoning_parser "$REASONING_PARSER"} )

#nohup ./experiments/quick_eval_bsub.sh --split base   --fg "${PARSER_ARG[@]}" "${ARGS[@]}" > output/run_base_${TS}.log   2>&1 &
nohup ./experiments/quick_eval_bsub.sh --split medium --fg "${PARSER_ARG[@]}" "${ARGS[@]}" > output/run_medium_${TS}.log 2>&1 &
nohup ./experiments/quick_eval_bsub.sh --split hard   --fg "${PARSER_ARG[@]}" "${ARGS[@]}" > output/run_hard_${TS}.log   2>&1 &

echo "Jobs submitted. Monitor progress with:"
echo "  tail -f output/run_medium_${TS}.log"
echo "  tail -f output/run_hard_${TS}.log"
