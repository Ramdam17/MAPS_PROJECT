#!/usr/bin/env bash
# Phase γ — Blindsight paper Table 5a Settings 4 (MAPS) and 5 (cascade 2nd only).
#
# Adds the two cells missing from the 2026-05-01 archive
# (knowyourself_outputs_20260501.tar.gz had only Settings 1, 2, 3, 6 via the
# legacy symmetric 2×2 factorial — see plan
# docs/plans/plan-20260519-blindsight-agl-settings-4-5-6.md §Problem).
#
# Uses the new 6-cell factorial (experiments/factorial_6cell) which expresses
# the asymmetric cascade required by paper Settings 4 and 5.
#
# Per-cell wall ≈ 4-5 s on Tamia CPU (smoke 2026-05-19 : BS S4 = 4.2 s,
# BS S5 = 3.6 s averaged over 5 seeds). 1000 cells (2 settings × 500 seeds)
# = ~80 min compute ; 4 h time budget = ~3× safety margin.
#
# Seed range : 42-541 (matches knowyourself_outputs_20260501.tar.gz convention,
# blindsight_array.sh § seq 42 541).
#
# Usage :
#   sbatch scripts/slurm/blindsight_settings_4_5.sh
#
# Outputs : $SCRATCH/maps/outputs/blindsight/<setting>/seed-<N>/{summary.json,...}.

#SBATCH --job-name=bs-s4s5
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-1%2                   # 2 tasks (one per new setting)
#SBATCH --time=04:00:00
#SBATCH --mem=4096M
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/blindsight-s4s5-%A_%a.out
#SBATCH --error=logs/slurm/blindsight-s4s5-%A_%a.err

set -euo pipefail

SETTINGS=(setting-4-maps setting-5-cascade-2nd)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SETTING=${SETTINGS[$TASK_ID]}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[bs-s4s5] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12

VENV="${REPO_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[bs-s4s5] ${VENV} missing — run 'uv sync --extra blindsight --extra dev' first." >&2
    exit 78
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1

echo "[bs-s4s5] task=${TASK_ID} setting=${SETTING} seeds=42..541"

# Seed loop. Skip cells whose summary.json already exists (idempotent re-run).
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/blindsight/${SETTING}"
mkdir -p "${OUT_BASE}"

for SEED in $(seq 42 541); do
    OUT_DIR="${OUT_BASE}/seed-${SEED}"
    if [[ -s "${OUT_DIR}/summary.json" ]]; then
        continue
    fi
    uv run --offline python scripts/run_blindsight.py \
        --factorial-config experiments/factorial_6cell \
        --setting "${SETTING}" \
        --seed "${SEED}"
done

# Verify completeness for this task's setting.
N_DONE=$(find "${OUT_BASE}" -name summary.json 2>/dev/null | wc -l)
echo "[bs-s4s5] task=${TASK_ID} setting=${SETTING} done=${N_DONE}/500"
if (( N_DONE < 500 )); then
    echo "[bs-s4s5] FAIL: incomplete — only ${N_DONE} cells written" >&2
    exit 1
fi
echo "[bs-s4s5] OK setting=${SETTING}"
