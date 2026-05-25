#!/usr/bin/env bash
# Phase F.2 — AGL reproduction on DRAC Tamia.
#
# Runs the full 2×2 factorial × 500 seeds (paper N=500, experiment_matrix.md,
# Tables 5b/5c). Full 3-phase protocol per scripts/run_agl.py :
# pretrain → replicate (20 networks) → train (high/low awareness) → evaluate.
#
# Per-seed wall ~10-30 s on CPU (small MLPs, lots of small ops). 500 seeds
# per setting × 4 settings = 2000 cells. Loop internally per setting.
#
# Usage :
#   sbatch scripts/slurm/agl_array.sh
#
# Outputs : $SCRATCH/maps/outputs/agl/<setting>/seed-<N>/summary.json + .pt + .npy.

#SBATCH --job-name=agl-array
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-3%4                   # 4 tasks (one per setting)
#SBATCH --time=06:00:00                 # generous : 500 cells × 30s = 4h ; buffer
#SBATCH --mem=4096M
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/agl-array-%A_%a.out
#SBATCH --error=logs/slurm/agl-array-%A_%a.err

set -euo pipefail

SETTINGS=(neither cascade_only second_order_only both)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SETTING=${SETTINGS[$TASK_ID]}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[agl] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12

VENV="${REPO_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[agl] ${VENV} missing — run 'uv sync --extra agl --extra dev' first." >&2
    exit 78
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1

echo "[agl] task=${TASK_ID} setting=${SETTING} seeds=42..541"

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/agl/${SETTING}"
mkdir -p "${OUT_BASE}"

for SEED in $(seq 42 541); do
    OUT_DIR="${OUT_BASE}/seed-${SEED}"
    if [[ -s "${OUT_DIR}/summary.json" ]]; then
        continue
    fi
    uv run --offline python scripts/run_agl.py \
        --setting "${SETTING}" \
        --seed "${SEED}"
done

N_DONE=$(find "${OUT_BASE}" -name summary.json 2>/dev/null | wc -l)
echo "[agl] task=${TASK_ID} setting=${SETTING} done=${N_DONE}/500"
if (( N_DONE < 500 )); then
    echo "[agl] FAIL: incomplete — only ${N_DONE} cells written" >&2
    exit 1
fi
echo "[agl] OK setting=${SETTING}"
