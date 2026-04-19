#!/usr/bin/env bash
# Sprint 07 Phase 4 — one-shot status of the SARL 300-cell array.
#
# Shows:
#   * queue state for the current user (all SARL-related jobs)
#   * per-game/setting completion counts
#   * full list of missing (game, setting, seed) cells
#
# Run from the project root:
#   bash scripts/slurm/monitor.sh
#
# Cells are considered "done" when their metrics.json exists and is
# non-empty under $SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<SEED>/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl"

GAMES=(breakout seaquest space_invaders asterix freeway)
SETTINGS=(1 2 3 4 5 6)
SEEDS=(42 43 44 45 46 47 48 49 50 51)

EXPECTED=$(( ${#GAMES[@]} * ${#SETTINGS[@]} * ${#SEEDS[@]} ))

echo "=== squeue --me ==="
squeue --me -o "%.10i %.9P %.12j %.2t %.10M %.10L %.6D %R" 2>&1 || true

echo
echo "=== outputs under ${OUT_BASE} ==="
if [[ ! -d "${OUT_BASE}" ]]; then
    echo "[monitor] output base does not exist yet — 0/${EXPECTED} cells"
    exit 0
fi

DONE=0
MISSING=()
for game in "${GAMES[@]}"; do
    for setting in "${SETTINGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cell="${game}/setting-${setting}/seed-${seed}"
            if [[ -s "${OUT_BASE}/${cell}/metrics.json" ]]; then
                DONE=$(( DONE + 1 ))
            else
                MISSING+=("${cell}")
            fi
        done
    done
done

echo "progress: ${DONE}/${EXPECTED} cells done"

# Per-game breakdown
echo
echo "per-game:"
for game in "${GAMES[@]}"; do
    c=0
    for setting in "${SETTINGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            [[ -s "${OUT_BASE}/${game}/setting-${setting}/seed-${seed}/metrics.json" ]] && c=$(( c + 1 ))
        done
    done
    printf "  %-15s %d/60\n" "${game}" "${c}"
done

if (( ${#MISSING[@]} > 0 )); then
    echo
    echo "missing (${#MISSING[@]}):"
    printf "  %s\n" "${MISSING[@]}"
fi
