#!/usr/bin/env bash
# End-to-end MedMNIST-Defer pipeline.
#
#   ./run_all.sh              # sample mode: committed data, a few minutes, no download
#   ./run_all.sh full         # full mode: downloads ~574 MB, reproduces the paper
set -euo pipefail

MODE="${1:-sample}"
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")"

echo "=== MedMNIST-Defer pipeline (mode: ${MODE}) ==="

if [ "${MODE}" = "full" ]; then
  echo
  echo "--- Step 1/5: download the full collection ---"
  "${PYTHON}" scripts/download_data.py
else
  echo
  echo "--- Step 1/5: refresh metadata manifest (skipped, using committed file) ---"
fi

echo
echo "--- Step 2/5: verify setup ---"
"${PYTHON}" scripts/verify_setup.py --mode "${MODE}"

echo
echo "--- Step 3/5: run benchmark ---"
"${PYTHON}" scripts/run_benchmark.py --mode "${MODE}"

echo
echo "--- Step 4/5: render figures ---"
"${PYTHON}" scripts/make_figures.py \
  --results "results/benchmark_${MODE}.json" \
  --outdir "results/figures_${MODE}"

echo
echo "--- Step 5/5: compare against reference results ---"
"${PYTHON}" scripts/compare_to_golden.py \
  --candidate "results/benchmark_${MODE}.json" || true

echo
echo "=== Done. Results in results/benchmark_${MODE}.json ==="
