#!/usr/bin/env bash
# Phase F.3 — SARL reproduction on DRAC Tamia (paper-faithful 500k frames).
#
# 90 cells = 5 games × 6 settings × 3 seeds (paper N=3, experiment_matrix.md
# Table 6). NOTE : Setting-7 (ACB, Young & Tian 2019) is a separate algorithm
# not implemented in run_sarl.py — run as a follow-up Phase F.5 if needed.
#
# Frames per cell : 500 000 (paper Table 11 explicit).
# Per-cell wall on H100 :
#   setting 1 (baseline)        ~30 min
#   setting 2 (cascade 1st)     ~2 h
#   setting 3 (meta only)       ~45 min
#   setting 4 (MAPS)            ~3 h
#   setting 5 (cascade 2nd)     ~2 h
#   setting 6 (full MAPS)       ~10-24 h  ← worst case, may need checkpoint
#
# Wall budget : 24 h max (Tamia partition cap). Setting-6 cells may not fit
# without checkpoint+resume — run_sarl.py has --resume / --resume-from since
# Sprint-08 D.14, so a requeued task picks up automatically.
#
# Concurrency : %4 (lab cap).
#
# Usage :
#   sbatch scripts/slurm/sarl_phase_f.sh                              # GPU H100 default
#   sbatch --export=DEVICE=cpu scripts/slurm/sarl_phase_f.sh          # CPU fallback

#SBATCH --job-name=sarl-phaseF
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-89%8                  # %8 : exceptional 4h boost ce soir si Guillaume est inactif ; SLURM auto-régule à %4 quand le cluster est plein
#SBATCH --time=24:00:00
#SBATCH --mem=8192M
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --requeue
#SBATCH --output=logs/slurm/sarl-phaseF-%A_%a.out
#SBATCH --error=logs/slurm/sarl-phaseF-%A_%a.err

set -euo pipefail

DEVICE=${DEVICE:-cuda}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[sarl-F] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12 cuda/12.6

VENV="${REPO_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[sarl-F] ${VENV} missing — run 'uv sync --extra sarl --extra dev' first." >&2
    exit 78
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1
# Tamia H100 = whole-node ; scope to GPU 0.
export CUDA_VISIBLE_DEVICES=0

# ── Task → (game, setting, seed) decode ────────────────────────────────────
GAMES=(breakout seaquest space_invaders asterix freeway)
SEEDS=(42 43 44)
N_SETTINGS=6
N_SEEDS=${#SEEDS[@]}
N_GAMES=${#GAMES[@]}
EXPECTED_TASKS=$(( N_GAMES * N_SETTINGS * N_SEEDS ))  # 90

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= EXPECTED_TASKS )); then
    echo "[sarl-F] TASK_ID=${TASK_ID} >= ${EXPECTED_TASKS} — out of range" >&2
    exit 64
fi

GAME_IDX=$(( TASK_ID / (N_SETTINGS * N_SEEDS) ))
REM=$(( TASK_ID % (N_SETTINGS * N_SEEDS) ))
SETTING=$(( REM / N_SEEDS + 1 ))
SEED_IDX=$(( REM % N_SEEDS ))

GAME=${GAMES[$GAME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[sarl-F] task=${TASK_ID} game=${GAME} setting=${SETTING} seed=${SEED} device=${DEVICE} frames=500000"

if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[sarl-F] no GPU visible — submit with --gpus-per-node=h100:4" >&2
        exit 64
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
fi

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl"
OUT_DIR="${OUT_BASE}/${GAME}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[sarl-F] ${OUT_DIR}/metrics.json already present — skip"
    exit 0
fi

# Use --resume so a requeued task picks up from the last checkpoint.
uv run --offline python scripts/run_sarl.py \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-frames 500000 \
    --output-dir "${OUT_DIR}" \
    --resume \
    -o "device=${DEVICE}"

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[sarl-F] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[sarl-F] OK cell=${GAME}/${SETTING}/${SEED}"
