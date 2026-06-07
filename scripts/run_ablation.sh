#!/usr/bin/env bash
# Four-condition ablation runner — prof's check-in (2026-05-31)
#
# Runs each of the 4 scoring methods 10 times on the Stage maze, records
# coverage + info-gain time series per trial, and dumps JSON for analysis.
#
# Usage (inside the ROS 2 / Stage Docker):
#   bash scripts/run_ablation.sh                   # all 4 methods × 10 trials
#   bash scripts/run_ablation.sh --trials 5        # quicker dry-run
#   bash scripts/run_ablation.sh --methods diffusion,heuristic   # subset
#
# Output: results/ablation/{method}/trial_{N}.json + summary CSV.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/results/ablation"
LAUNCH="${PROJECT_DIR}/ros2_ws/src/diffusion_explorer/launch_exploration.sh"

TRIALS=10
METHODS="nearest,info_gain,heuristic,diffusion"
TIMEOUT_S=180        # max wall-clock seconds per trial (covers reach to 85%)
COVERAGE_GOAL=0.85

while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)   TRIALS="$2"; shift 2 ;;
        --methods)  METHODS="$2"; shift 2 ;;
        --timeout)  TIMEOUT_S="$2"; shift 2 ;;
        *)          echo "unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"
IFS=',' read -ra METHOD_ARR <<< "$METHODS"

echo "=========================================="
echo " ablation run"
echo "=========================================="
echo "  methods: ${METHODS}"
echo "  trials:  ${TRIALS}"
echo "  timeout: ${TIMEOUT_S}s per trial"
echo "  out:     ${OUT_DIR}"
echo ""

for method in "${METHOD_ARR[@]}"; do
    mkdir -p "${OUT_DIR}/${method}"
    echo "----- method: ${method} -----"
    for ((n=1; n<=TRIALS; n++)); do
        TRIAL_LOG="${OUT_DIR}/${method}/trial_${n}.log"
        TRIAL_JSON="${OUT_DIR}/${method}/trial_${n}.json"

        echo "  trial ${n}/${TRIALS}..."
        # Each trial: launch the stack with this method, record /map coverage
        # + chosen frontier info-gain to a JSON line per scoring decision.
        # The exploration_manager already exits when coverage_threshold is met;
        # we set it via env so analysis is reproducible.
        COVERAGE_THRESHOLD="${COVERAGE_GOAL}" \
        TRIAL_OUT_JSON="${TRIAL_JSON}" \
        timeout "${TIMEOUT_S}s" bash "${LAUNCH}" --method "${method}" \
            > "${TRIAL_LOG}" 2>&1 || true
        echo "    saved → ${TRIAL_JSON}"
    done
done

echo ""
echo "=========================================="
echo " analysis"
echo "=========================================="
python3 "${SCRIPT_DIR}/analyze_ablation.py" --ablation_dir "${OUT_DIR}"
echo ""
echo "done. inspect:"
echo "  ${OUT_DIR}/summary.csv"
echo "  ${OUT_DIR}/coverage_curves.png"
echo "  ${OUT_DIR}/info_gain_bar.png"
