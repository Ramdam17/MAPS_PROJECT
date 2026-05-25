#!/usr/bin/env bash
# Phase F.4 — SARL+CL reproduction on DRAC Tamia.
#
# 90 cells = 5 games × 6 settings × 3 seeds (paper N=3, paper Tables 6 + CL
# columns). Same factorial as F.3 but with curriculum learning enabled.
#
# Frames per cell : 500 000 base × 2 (paper §B.3 explicit "double when
# training with our curriculum learning approach"). Per-cell wall on H100
# is approximately 2× the corresponding F.3 cell.
#
# Usage :
#   sbatch scripts/slurm/sarl_cl_phase_f.sh                            # GPU H100 default
#   sbatch --dependency=afterany:<sarl_phase_f_jobid> ...              # chain after F.3

#SBATCH --job-name=sarl-cl-phaseF
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-89%8                  # %8 : 24h-boost lab autorisé ; SLURM auto-régule selon dispo H100
#SBATCH --time=24:00:00
#SBATCH --mem=8192M
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --requeue
#SBATCH --output=logs/slurm/sarl-cl-phaseF-%A_%a.out
#SBATCH --error=logs/slurm/sarl-cl-phaseF-%A_%a.err

set -euo pipefail

DEVICE=${DEVICE:-cuda}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[sarl-cl-F] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12 cuda/12.6

VENV="${REPO_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[sarl-cl-F] ${VENV} missing — run 'uv sync --extra sarl --extra dev' first." >&2
    exit 78
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

GAMES=(breakout seaquest space_invaders asterix freeway)
SEEDS=(42 43 44)
N_SETTINGS=6
N_SEEDS=${#SEEDS[@]}
N_GAMES=${#GAMES[@]}
EXPECTED_TASKS=$(( N_GAMES * N_SETTINGS * N_SEEDS ))  # 90

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= EXPECTED_TASKS )); then
    echo "[sarl-cl-F] TASK_ID=${TASK_ID} out of range" >&2
    exit 64
fi

GAME_IDX=$(( TASK_ID / (N_SETTINGS * N_SEEDS) ))
REM=$(( TASK_ID % (N_SETTINGS * N_SEEDS) ))
SETTING=$(( REM / N_SEEDS + 1 ))
SEED_IDX=$(( REM % N_SEEDS ))

GAME=${GAMES[$GAME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[sarl-cl-F] task=${TASK_ID} game=${GAME} setting=${SETTING} seed=${SEED} device=${DEVICE} frames=1000000 (CL ×2)"

if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[sarl-cl-F] no GPU visible — submit with --gpus-per-node=h100:4" >&2
        exit 64
    fi
fi

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl_cl"
OUT_DIR="${OUT_BASE}/${GAME}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[sarl-cl-F] ${OUT_DIR}/metrics.json already present — skip"
    exit 0
fi

uv run --offline python scripts/run_sarl_cl.py \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-frames 1000000 \
    --output-dir "${OUT_DIR}" \
    --resume \
    -o "device=${DEVICE}"

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[sarl-cl-F] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[sarl-cl-F] OK cell=${GAME}/${SETTING}/${SEED}"
