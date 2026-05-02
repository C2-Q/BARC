#!/usr/bin/env bash
# One-command reproduction for the workload-family extension on branch
# `add-real-benchmark-families`.
#
# Regenerates:
#   - per-instance result CSVs under outputs/real_trace/
#   - aggregate tables under outputs/tables/
#       real_trace_workload_families_raw.csv
#       real_trace_workload_families_representative_summary.csv
#       real_trace_workload_families_scaling_summary.csv
#       qaoa_summary.csv
#   - figures at outputs/figures/qft_vs_real_traces.{png,pdf}
#   - figures at outputs/figures/real_trace_scaling.{png,pdf}
#
# This script does NOT regenerate:
#   - constructed-DAG (compressibility-family) experiments
#   - lower-bound validation tables (e.g. lower_bound_validation.csv)
#   - any other figure produced by run_qce_paper.py
#
# It assumes the existing main pipeline has already produced
# `outputs/tables/real_trace_scaling_summary.csv` and
# `outputs/tables/qft_real_trace_summary.csv`. If they are missing, run the
# main pipeline first (see README "Reproduce the paper figures").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${STAGE1_DIR}"

PYTHON="${PYTHON:-python}"

echo "[1/2] Running workload-family driver (CLA, modular-arithmetic block, QAOA MaxCut)..."
"${PYTHON}" scripts/run_real_trace_workload_families.py

echo "[2/2] Rebuilding qft_vs_real_traces and real_trace_scaling figures..."
"${PYTHON}" scripts/build_workload_family_figures.py

echo
echo "Done. New artefacts:"
echo "  outputs/tables/real_trace_workload_families_raw.csv"
echo "  outputs/tables/real_trace_workload_families_representative_summary.csv"
echo "  outputs/tables/real_trace_workload_families_scaling_summary.csv"
echo "  outputs/tables/qaoa_summary.csv"
echo "  outputs/figures/qft_vs_real_traces.{png,pdf}"
echo "  outputs/figures/real_trace_scaling.{png,pdf}"
