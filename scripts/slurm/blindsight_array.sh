#!/usr/bin/env bash
# Phase F.1 — Blindsight reproduction on DRAC Tamia.
#
# Runs the missing settings of the 2×2 factorial × 500 seeds (paper N=500,
# experiment_matrix.md, Table 5a). Existing scratch already has ``both`` and
# ``neither`` × seeds 42-541 ; this array fills in the remaining two
# settings : ``cascade_only`` and ``second_order_only``.
#
# Per-cell wall ≈ 7 s (measured from existing summary.json elapsed_seconds).
# 1000 cells (2 settings × 500 seeds) = ~2 h compute.
# Run as a SLURM array : 2 tasks, each loops over its 500 seeds in Python
# (cells are too short to justify per-cell SLURM dispatch overhead).
#
# Usage :
#   sbatch scripts/slurm/blindsight_array.sh
#
# Outputs : $SCRATCH/maps/outputs/blindsight/<setting>/seed-<N>/{summary.json,...}.

#SBATCH --job-name=blindsight-array
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-1%2                   # 2 tasks (one per missing setting)
#SBATCH --time=04:00:00                 # 500 cells × ~7s = ~1h ; buffer ×4
#SBATCH --mem=4096M
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/blindsight-array-%A_%a.out
#SBATCH --error=logs/slurm/blindsight-array-%A_%a.err

set -euo pipefail

SETTINGS=(cascade_only second_order_only)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SETTING=${SETTINGS[$TASK_ID]}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[blindsight] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12

VENV="${REPO_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[blindsight] ${VENV} missing — run 'uv sync --extra blindsight --extra dev' first." >&2
    exit 78
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1

echo "[blindsight] task=${TASK_ID} setting=${SETTING} seeds=42..541"

# Seed loop. Skip cells whose summary.json already exists (idempotent re-run).
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/blindsight/${SETTING}"
mkdir -p "${OUT_BASE}"

for SEED in $(seq 42 541); do
    OUT_DIR="${OUT_BASE}/seed-${SEED}"
    if [[ -s "${OUT_DIR}/summary.json" ]]; then
        continue
    fi
    uv run --offline python scripts/run_blindsight.py \
        --setting "${SETTING}" \
        --seed "${SEED}"
done

# Verify completeness for this task's setting.
N_DONE=$(find "${OUT_BASE}" -name summary.json 2>/dev/null | wc -l)
echo "[blindsight] task=${TASK_ID} setting=${SETTING} done=${N_DONE}/500"
if (( N_DONE < 500 )); then
    echo "[blindsight] FAIL: incomplete — only ${N_DONE} cells written" >&2
    exit 1
fi
echo "[blindsight] OK setting=${SETTING}"
