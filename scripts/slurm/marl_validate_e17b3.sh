#!/usr/bin/env bash
# E.17b3 — 3-seed paper-validation on commons_harvest_closed × baseline × 300k.
#
# Reproduces the paper's "Setting-1 (Baseline)" row of Table 7 for
# commons_harvest_closed. Target : mean episode reward over the last ~50
# eps should fall near **19.52 ± 0.71** across the 3 seeds.
#
# Cells : 3 seeds (42, 43, 44) × 1 substrate × 1 setting = 3 parallel tasks.
# Per-cell budget : 300k steps × 300 eps × ~18 s/ep ≈ 1 h 30 min.
# Wall budget : 2 h (safety margin) × 3 parallel → total wall ~2 h with %3.
#
# Usage :
#   sbatch scripts/slurm/marl_validate_e17b3.sh
#
# Outputs land in $SCRATCH/maps/outputs/marl/commons_harvest_closed/
# setting-baseline/seed-{42,43,44}-validate/metrics.json.

#SBATCH --job-name=marl-validate
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-2%3
#SBATCH --time=02:00:00
#SBATCH --mem=16384M
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=logs/slurm/marl-validate-%A_%a.out
#SBATCH --error=logs/slurm/marl-validate-%A_%a.err

set -euo pipefail

SUBSTRATE=commons_harvest_closed
SETTING=baseline
NUM_ENV_STEPS=300000
SEEDS=(42 43 44)

SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[marl-validate] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.11.5 cuda/12.6

VENV_MARL="${REPO_ROOT}/.venv-marl"
if [[ ! -x "${VENV_MARL}/bin/python" ]]; then
    echo "[marl-validate] ${VENV_MARL} missing — run install per docs/install_marl_drac.md first." >&2
    exit 78
fi

export VIRTUAL_ENV="${VENV_MARL}"
export PATH="${VENV_MARL}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "[marl-validate] task=${SLURM_ARRAY_TASK_ID} substrate=${SUBSTRATE} setting=${SETTING} seed=${SEED} num_env_steps=${NUM_ENV_STEPS}"
echo "[marl-validate] python=$(python --version 2>&1)"

if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[marl-validate] no GPU visible — did the allocation include --gpus-per-node=h100:4 ?" >&2
    exit 64
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl"
OUT_DIR="${OUT_BASE}/${SUBSTRATE}/setting-${SETTING}/seed-${SEED}-validate"
mkdir -p "${OUT_DIR}"
echo "[marl-validate] out_dir=${OUT_DIR}"

if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-validate] ${OUT_DIR}/metrics.json already present — skip"
    exit 0
fi

time python scripts/run_marl.py \
    --substrate "${SUBSTRATE}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --device cuda \
    --num-env-steps "${NUM_ENV_STEPS}" \
    --output-dir "${OUT_DIR}" \
    --resume

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-validate] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[marl-validate] OK seed=${SEED} — metrics persisted ($(stat -c %s "${OUT_DIR}/metrics.json") bytes)"
