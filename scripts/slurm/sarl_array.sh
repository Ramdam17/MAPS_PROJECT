#!/usr/bin/env bash
# Sprint 07 — SARL reproduction on Compute Canada Narval.
#
# Submits 150 DQN training cells as a SLURM array:
#   games    = {breakout, seaquest, space_invaders, asterix, freeway}    (5)
#   settings = {1, 2, 3, 4, 5, 6}                                         (6)
#   seeds    = {42, 43, 44, 45, 46}                                       (5)
#   total    = 5 × 6 × 5 = 150 cells
#
# Each array task runs `scripts/run_sarl.py` for one (game, setting, seed)
# cell at the paper's 5M-frame horizon. Outputs land in
# $SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<SEED>/ which a
# post-array sbatch (`aggregate.sh`) then picks up.
#
# Submit with:
#   sbatch --account=<rrg-group> scripts/slurm/sarl_array.sh
#
# Tune --time, --gres, --mem based on actual per-cell wall-clock from
# Sprint 07 §7.1 smoke test calibration.

#SBATCH --job-name=maps-sarl
#SBATCH --array=0-149%20               # 150 tasks, up to 20 concurrent
#SBATCH --time=24:00:00                # per-task — revise after smoke test
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/maps-sarl-%A_%a.out
#SBATCH --error=logs/slurm/maps-sarl-%A_%a.err

set -euo pipefail

# ── Task → (game, setting, seed) decode ────────────────────────────────────
GAMES=(breakout seaquest space_invaders asterix freeway)
SEEDS=(42 43 44 45 46)
N_SETTINGS=6
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
GAME_IDX=$(( TASK_ID / (N_SETTINGS * N_SEEDS) ))
REM=$(( TASK_ID % (N_SETTINGS * N_SEEDS) ))
SETTING=$(( REM / N_SEEDS + 1 ))
SEED_IDX=$(( REM % N_SEEDS ))

GAME=${GAMES[$GAME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Task $TASK_ID → game=$GAME setting=$SETTING seed=$SEED"

# ── Environment (Narval-specific — see docs/install_linux.md) ──────────────
module load python/3.12
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# uv is lab-standard; activate the pre-synced venv
source .venv/bin/activate

OUTDIR="${SCRATCH:-$PWD}/maps/outputs/sarl/$GAME/setting-$SETTING/seed-$SEED"
mkdir -p "$OUTDIR"

# ── Run ────────────────────────────────────────────────────────────────────
python scripts/run_sarl.py \
    --game "$GAME" \
    --setting "$SETTING" \
    --seed "$SEED" \
    -o "device=cuda" \
    -o "output_dir=$OUTDIR"

echo "Cell complete: $OUTDIR"
