#!/usr/bin/env bash
# Phase F.4 — SARL+CL reproduction on DRAC Narval (A100 GPU, def-gdumas85_gpu).
#
# Counterpart to scripts/slurm/sarl_cl_phase_f.sh which runs on Tamia (H100).
# Narval-specific deltas:
#   --account=def-gdumas85_gpu         (Narval splits CPU/GPU allocations)
#   --gres=gpu:a100:1                  (Narval allows per-GPU; Tamia is whole-node)
#   --time=72:00:00                    (Narval allows up to 7 days; 72h headroom for setting 6)
#   module cuda/12.2                   (matches torch cu121 wheel runtime)
#   venv .venv-narval                  (separate from Tamia .venv with cu128 pin)
#
# Same factorial as Tamia F.4: 90 cells = 5 games × 6 settings × 3 seeds × 1M frames
# (paper §B.3 doubles base 500k for CL). Idempotent metrics.json guard + --resume
# to handle pre-emption.
#
# Usage:
#   sbatch scripts/slurm/sarl_cl_narval.sh                                    # GPU A100
#   sbatch --array=0%1 scripts/slurm/sarl_cl_narval.sh                        # smoke (1 cell)
#   sbatch --array=0-89%8 scripts/slurm/sarl_cl_narval.sh                     # boost concurrency post smoke

#SBATCH --job-name=sarl-cl-narval
#SBATCH --account=def-gdumas85_gpu
#SBATCH --array=0-89%4
#SBATCH --time=72:00:00
#SBATCH --mem=8192M
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/sarl-cl-narval-%A_%a.out
#SBATCH --error=logs/slurm/sarl-cl-narval-%A_%a.err

set -euo pipefail

DEVICE=${DEVICE:-cuda}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[sarl-cl-narval] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12 cuda/12.2

VENV="${REPO_ROOT}/.venv-narval"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[sarl-cl-narval] ${VENV} missing — run venv setup on the login node first." >&2
    echo "[sarl-cl-narval] Expected layout: $SCRATCH/maps/venv-narval + symlink .venv-narval" >&2
    exit 78
fi

export PYTHONUNBUFFERED=1
# Direct python invocation below — bypass `uv run` which would re-resolve
# pyproject.toml deps (cu128 pin) and override our hand-built cu121 venv.

GAMES=(breakout seaquest space_invaders asterix freeway)
SEEDS=(42 43 44)
N_SETTINGS=6
N_SEEDS=${#SEEDS[@]}
N_GAMES=${#GAMES[@]}
EXPECTED_TASKS=$(( N_GAMES * N_SETTINGS * N_SEEDS ))  # 90

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= EXPECTED_TASKS )); then
    echo "[sarl-cl-narval] TASK_ID=${TASK_ID} out of range" >&2
    exit 64
fi

GAME_IDX=$(( TASK_ID / (N_SETTINGS * N_SEEDS) ))
REM=$(( TASK_ID % (N_SETTINGS * N_SEEDS) ))
SETTING=$(( REM / N_SEEDS + 1 ))
SEED_IDX=$(( REM % N_SEEDS ))

GAME=${GAMES[$GAME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[sarl-cl-narval] task=${TASK_ID} game=${GAME} setting=${SETTING} seed=${SEED} device=${DEVICE} frames=1000000 (CL ×2)"

if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[sarl-cl-narval] no GPU visible — submit with --gres=gpu:a100:1" >&2
        exit 64
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
fi

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl_cl"
OUT_DIR="${OUT_BASE}/${GAME}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[sarl-cl-narval] ${OUT_DIR}/metrics.json already present — skip"
    exit 0
fi

"${VENV}/bin/python" scripts/run_sarl_cl.py \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-frames 1000000 \
    --output-dir "${OUT_DIR}" \
    --resume \
    -o "device=${DEVICE}"

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[sarl-cl-narval] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[sarl-cl-narval] OK cell=${GAME}/${SETTING}/${SEED}"
