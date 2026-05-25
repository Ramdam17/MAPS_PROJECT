#!/usr/bin/env bash
# E.17b — MARL mini-bench on DRAC Tamia H100.
#
# Single cell, 1 M env steps (10× the E.16 smoke). Validates :
# - No NaN / crash over a sustained run.
# - Checkpoint writes every ``save_interval`` episodes and on completion.
# - Convergence signals (value_loss trend, entropy decay, ratio stability)
#   to refine the full-run ETA before E.17c.
# - Wall-time estimate for the 300 k × 72-cell array.
#
# Budget : ~5-6 h expected (100 k took 31 min ; linear scaling to 1 M = 5.2 h).
# Reserve 8 h for safety — checkpoint lets us resume if we blow the wall.

#SBATCH --job-name=marl-bench
#SBATCH --account=aip-gdumas85
#SBATCH --time=08:00:00
#SBATCH --mem=16384M
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=logs/slurm/marl-bench-%j.out
#SBATCH --error=logs/slurm/marl-bench-%j.err

set -euo pipefail

SUBSTRATE=${1:-commons_harvest_closed}
SETTING=${2:-baseline}
SEED=${3:-42}
NUM_ENV_STEPS=${4:-1000000}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[marl-bench] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.11.5 cuda/12.6

VENV_MARL="${REPO_ROOT}/.venv-marl"
if [[ ! -x "${VENV_MARL}/bin/python" ]]; then
    echo "[marl-bench] ${VENV_MARL} missing — run install per docs/install_marl_drac.md first." >&2
    exit 78
fi

export VIRTUAL_ENV="${VENV_MARL}"
export PATH="${VENV_MARL}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "[marl-bench] substrate=${SUBSTRATE} setting=${SETTING} seed=${SEED} num_env_steps=${NUM_ENV_STEPS}"
echo "[marl-bench] python=$(python --version 2>&1)"

if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[marl-bench] no GPU visible — did the allocation include --gpus-per-node=h100:4 ?" >&2
    exit 64
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl"
OUT_DIR="${OUT_BASE}/${SUBSTRATE}/setting-${SETTING}/seed-${SEED}-bench1M"
mkdir -p "${OUT_DIR}"
echo "[marl-bench] out_dir=${OUT_DIR}"

# Auto-resume : --resume is a silent no-op if no checkpoint exists, so it's
# always safe to pass. Protects us if the allocation is requeued.
time python scripts/run_marl.py \
    --substrate "${SUBSTRATE}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --device cuda \
    --num-env-steps "${NUM_ENV_STEPS}" \
    --output-dir "${OUT_DIR}" \
    --resume

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-bench] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[marl-bench] OK — metrics persisted ($(stat -c %s "${OUT_DIR}/metrics.json") bytes)"
