#!/usr/bin/env bash
# Phase F.4 smoke — single SARL+CL cell on Narval to validate venv + GPU + I/O
# before submitting the 90-cell array.
#
# Defaults: breakout setting 1 seed 42 50000 frames cuda → ~5 min on A100.
# Writes metrics.json under $SCRATCH/maps/outputs/sarl_cl-smoke/.
#
# Usage:
#   sbatch scripts/slurm/smoke_sarl_cl_narval.sh                                       # all defaults
#   sbatch scripts/slurm/smoke_sarl_cl_narval.sh breakout 1 42 50000 cuda

#SBATCH --job-name=sarl-cl-narval-smoke
#SBATCH --account=def-gdumas85_gpu
#SBATCH --time=01:00:00
#SBATCH --mem=8192M
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/slurm/sarl-cl-narval-smoke-%j.out
#SBATCH --error=logs/slurm/sarl-cl-narval-smoke-%j.err

set -euo pipefail

GAME=${1:-breakout}
SETTING=${2:-1}
SEED=${3:-42}
N_FRAMES=${4:-50000}
DEVICE=${5:-cuda}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[smoke] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12 cuda/12.2

VENV="${REPO_ROOT}/.venv-narval"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[smoke] ${VENV} missing — run venv setup on the login node first." >&2
    exit 78
fi

export PYTHONUNBUFFERED=1

if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[smoke] DEVICE=cuda but no GPU visible — submit with --gres=gpu:a100:1" >&2
        exit 64
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    "${VENV}/bin/python" -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
fi

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl_cl-smoke"
OUT_DIR="${OUT_BASE}/${GAME}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

echo "[smoke] game=${GAME} setting=${SETTING} seed=${SEED} frames=${N_FRAMES} device=${DEVICE}"
echo "[smoke] out_dir=${OUT_DIR}"

time "${VENV}/bin/python" scripts/run_sarl_cl.py \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-frames "${N_FRAMES}" \
    --output-dir "${OUT_DIR}" \
    -o "device=${DEVICE}"

METRICS="${OUT_DIR}/metrics.json"
if [[ ! -s "${METRICS}" ]]; then
    echo "[smoke] FAIL: ${METRICS} missing or empty" >&2
    exit 1
fi
echo "[smoke] OK — metrics persisted ($(stat -c %s "${METRICS}") bytes)"
