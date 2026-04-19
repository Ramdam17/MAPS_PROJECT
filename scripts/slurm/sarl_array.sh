#!/usr/bin/env bash
# Sprint 07 Phase 4 — SARL reproduction array on Tamia.
#
# 300 cells = 5 games × 6 settings × 10 seeds (paper-conforming N=10).
#   games    = {breakout, seaquest, space_invaders, asterix, freeway}
#   settings = {1..6}
#   seeds    = {42..51}
#
# Per-cell resources are CONSERVATIVE DEFAULTS; they MUST be re-calibrated
# from Phase 2 bench numbers before the full run. Override at submit time.
#
#   # CPU path (default, safe on Tamia without GPU contention):
#   sbatch scripts/slurm/sarl_array.sh
#
#   # GPU path (decided only if Phase 2 shows ≥ 1.5× CPU speedup):
#   sbatch --gpus-per-node=h100:4 --cpus-per-task=4 --mem=16000M --time=04:00:00 \
#          --export=DEVICE=cuda scripts/slurm/sarl_array.sh
#
# Outputs land in $SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<SEED>/
# and are rsync'd to outputs/sarl/ by scripts/slurm/aggregate.sh.

#SBATCH --job-name=sarl-array
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-299%10                # 300 cells, 10 concurrent — widen after Phase 2
#SBATCH --time=06:00:00                 # per-task; calibrate = 1.1 × bench_wall_500k × 10
#SBATCH --mem=8000M                     # calibrate from peak_rss_mb + margin
#SBATCH --cpus-per-task=4
#SBATCH --requeue                       # survive pre-emption
#SBATCH --output=logs/slurm/sarl-array-%A_%a.out
#SBATCH --error=logs/slurm/sarl-array-%A_%a.err

set -euo pipefail

# Device: cpu (default) or cuda. Override with `--export=DEVICE=cuda` at submit.
DEVICE=${DEVICE:-cpu}

# shellcheck source=scripts/slurm/common.sh
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/common.sh"

# ── Task → (game, setting, seed) decode ────────────────────────────────────
GAMES=(breakout seaquest space_invaders asterix freeway)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
N_SETTINGS=6
N_SEEDS=${#SEEDS[@]}
N_GAMES=${#GAMES[@]}
EXPECTED_TASKS=$(( N_GAMES * N_SETTINGS * N_SEEDS ))  # 300

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= EXPECTED_TASKS )); then
    echo "[array] TASK_ID=${TASK_ID} >= ${EXPECTED_TASKS} — out of range" >&2
    exit 64
fi

GAME_IDX=$(( TASK_ID / (N_SETTINGS * N_SEEDS) ))
REM=$(( TASK_ID % (N_SETTINGS * N_SEEDS) ))
SETTING=$(( REM / N_SEEDS + 1 ))
SEED_IDX=$(( REM % N_SEEDS ))

GAME=${GAMES[$GAME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[array] task=${TASK_ID} game=${GAME} setting=${SETTING} seed=${SEED} device=${DEVICE}"

# ── GPU sanity ─────────────────────────────────────────────────────────────
if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[array] DEVICE=cuda but no GPU visible — submit with --gpus-per-node=h100:4" >&2
        exit 64
    fi
fi

# ── Output dir — $SCRATCH on DRAC ──────────────────────────────────────────
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl"
OUT_DIR="${OUT_BASE}/${GAME}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

# Idempotent re-run guard: if metrics.json already exists and is non-empty,
# this cell is done — skip (useful after --requeue).
if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[array] ${OUT_DIR}/metrics.json already present — skip"
    exit 0
fi

# ── Run ────────────────────────────────────────────────────────────────────
uv run --offline python scripts/run_sarl.py \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    -o "device=${DEVICE}"

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[array] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[array] OK cell=${GAME}/${SETTING}/${SEED}"
