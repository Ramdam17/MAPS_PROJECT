#!/usr/bin/env bash
# Sprint 07 Phase 2 — SARL benchmark (1 cell × 500k frames).
#
# Purpose: measure frames/s, updates/s, peak VRAM, peak RSS on the target
# partition so we can calibrate --time/--mem for the 300-cell array
# and decide CPU vs GPU.
#
# Usage:
#   sbatch scripts/slurm/bench_sarl.sh MODE [GAME] [SETTING] [SEED] [N_FRAMES]
#
#   MODE ∈ {cpu_4c, gpu_full}
#     cpu_4c   — --cpus-per-task=4, device=cpu
#     gpu_full — --gpus-per-node=h100:4, device=cuda
#                (Tamia allocates H100 per whole node; no MIG available)
#
# Defaults: breakout 6 42 500000.
#
# Writes: $SCRATCH/maps/bench/bench-<mode>-s<SETTING>-seed<SEED>-<git_sha>.json
#         (falls back to ./outputs/bench/... when $SCRATCH is unset — dev boxes).

#SBATCH --job-name=sarl-bench
#SBATCH --account=aip-gdumas85
#SBATCH --time=02:00:00
#SBATCH --mem=16000M
#SBATCH --output=logs/slurm/sarl-bench-%j.out
#SBATCH --error=logs/slurm/sarl-bench-%j.err

# NOTE — MODE-specific resources must be passed at submit time:
#   sbatch --cpus-per-task=4              scripts/slurm/bench_sarl.sh cpu_4c
#   sbatch --gpus-per-node=h100:4 --cpus-per-task=4 scripts/slurm/bench_sarl.sh gpu_full
#
# We can't easily conditionalize --gres in #SBATCH, and Tamia rejects
# --gpus-per-node when running CPU-only, so we keep both as submit-time flags.

set -euo pipefail

MODE=${1:?"usage: sbatch [--cpus-per-task=N --gpus-per-node=...] bench_sarl.sh MODE [GAME] [SETTING] [SEED] [N_FRAMES]"}
GAME=${2:-breakout}
SETTING=${3:-6}
SEED=${4:-42}
N_FRAMES=${5:-500000}

# shellcheck source=scripts/slurm/common.sh
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/common.sh"

case "${MODE}" in
    cpu_4c)
        DEVICE=cpu
        ;;
    gpu_full)
        DEVICE=cuda
        if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            echo "[bench] MODE=gpu_full but no GPU visible — resubmit with --gpus-per-node=h100:4" >&2
            exit 64
        fi
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
        ;;
    *)
        echo "[bench] unknown MODE=${MODE} — expected cpu_4c or gpu_full" >&2
        exit 64
        ;;
esac

OUT_DIR="${SCRATCH:-${REPO_ROOT}/outputs}/maps/bench"
mkdir -p "${OUT_DIR}"

echo "[bench] MODE=${MODE} DEVICE=${DEVICE} game=${GAME} setting=${SETTING} seed=${SEED} frames=${N_FRAMES}"

time uv run --offline python scripts/bench_sarl.py \
    --mode "${MODE}" \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-frames "${N_FRAMES}" \
    --device "${DEVICE}" \
    --output-dir "${OUT_DIR}"

# Show result summary in the slurm log
ls -la "${OUT_DIR}"
