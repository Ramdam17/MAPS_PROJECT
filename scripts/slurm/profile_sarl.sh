#!/usr/bin/env bash
# Sprint 07 Phase 2.C — cProfile a single SARL training cell, dump top-40.
#
# Usage:
#   sbatch scripts/slurm/profile_sarl.sh [GAME] [SETTING] [SEED] [N_FRAMES]
#
# Defaults: breakout 6 42 25000 (setting 6 at 7 fps CPU ≈ 1h wall).
# Writes:
#   $SCRATCH/maps/bench/prof-<setting>-<seed>-<rev>.out   (cProfile binary dump)
#   $SCRATCH/maps/bench/prof-<setting>-<seed>-<rev>.txt   (top-40 cumulative+tottime)
#   (falls back to ./outputs/bench/... when $SCRATCH is unset — dev boxes).

#SBATCH --job-name=sarl-profile
#SBATCH --account=aip-gdumas85
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/sarl-profile-%j.out
#SBATCH --error=logs/slurm/sarl-profile-%j.err

set -euo pipefail

GAME=${1:-breakout}
SETTING=${2:-6}
SEED=${3:-42}
N_FRAMES=${4:-25000}

# shellcheck source=scripts/slurm/common.sh
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/common.sh"

REV=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)
OUT_DIR="${SCRATCH:-${REPO_ROOT}/outputs}/maps/bench"
mkdir -p "${OUT_DIR}"
PROF_BIN="${OUT_DIR}/prof-${SETTING}-${SEED}-${REV}.out"
PROF_TXT="${OUT_DIR}/prof-${SETTING}-${SEED}-${REV}.txt"

# Use a throwaway output dir for run_sarl metrics (not the point here).
RUN_OUT="${OUT_DIR}/_profile-run-${SETTING}-${SEED}"
mkdir -p "${RUN_OUT}"

echo "[profile] game=${GAME} setting=${SETTING} seed=${SEED} frames=${N_FRAMES}"
echo "[profile] prof_bin=${PROF_BIN}"

# cProfile wraps run_sarl.py. Device=cpu (profile is CPU-thread time).
time uv run --offline python -m cProfile -o "${PROF_BIN}" scripts/run_sarl.py \
    --game "${GAME}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-frames "${N_FRAMES}" \
    --output-dir "${RUN_OUT}" \
    -o "device=cpu"

uv run --offline python scripts/analyze_profile.py "${PROF_BIN}" 40 > "${PROF_TXT}"
echo "[profile] wrote ${PROF_TXT} ($(wc -l < "${PROF_TXT}") lines)"
head -30 "${PROF_TXT}" || true
