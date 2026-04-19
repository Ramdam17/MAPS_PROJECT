#!/usr/bin/env bash
# Sprint 07 — SARL smoke test (1 cell, 50k frames, CPU or GPU).
#
# Validates the full pipeline under SLURM before any benchmark or array:
# one (game, setting, seed) cell trains end-to-end, writes metrics.json,
# exits clean. Completes in ≲20 min CPU, ≲10 min GPU on Tamia.
#
# Usage:
#   sbatch scripts/slurm/smoke_sarl.sh [GAME] [SETTING] [SEED] [N_FRAMES] [DEVICE]
#
# Defaults: breakout 6 42 50000 cpu.
# DEVICE ∈ {cpu, cuda}.
#
# Outputs: $SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<SEED>/metrics.json
#          (falls back to outputs/sarl/... if $SCRATCH is unset)

#SBATCH --job-name=sarl-smoke
#SBATCH --account=aip-gdumas85
#SBATCH --time=01:00:00
#SBATCH --mem=8000M
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/sarl-smoke-%j.out
#SBATCH --error=logs/slurm/sarl-smoke-%j.err

# Positional args (before common.sh — it does `set -u`)
GAME=${1:-breakout}
SETTING=${2:-6}
SEED=${3:-42}
N_FRAMES=${4:-50000}
DEVICE=${5:-cpu}

# shellcheck source=scripts/slurm/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ── GPU sanity (Tamia H100 = whole-node allocation) ────────────────────────
if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[smoke] DEVICE=cuda but no GPU visible — resubmit with --gpus-per-node=h100:4" >&2
        exit 64  # EX_USAGE
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
fi

# ── Output dir — $SCRATCH on DRAC, project tree otherwise ──────────────────
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl"
OUT_DIR="${OUT_BASE}/${GAME}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

echo "[smoke] game=${GAME} setting=${SETTING} seed=${SEED} frames=${N_FRAMES} device=${DEVICE}"
echo "[smoke] out_dir=${OUT_DIR}"

# ── Run ────────────────────────────────────────────────────────────────────
time uv run --offline python scripts/run_sarl.py \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-frames "${N_FRAMES}" \
    --output-dir "${OUT_DIR}" \
    -o "device=${DEVICE}"

# ── Verify metrics ─────────────────────────────────────────────────────────
METRICS="${OUT_DIR}/metrics.json"
if [[ ! -s "${METRICS}" ]]; then
    echo "[smoke] FAIL: ${METRICS} missing or empty" >&2
    exit 1
fi
echo "[smoke] OK — metrics persisted ($(stat -c %s "${METRICS}") bytes)"
