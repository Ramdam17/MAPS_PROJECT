#!/usr/bin/env bash
# Sprint 07 Phase 5 — post-array aggregation on Tamia.
#
# Rsyncs 300 cells from $SCRATCH into the project tree, then runs
# scripts/aggregate_sarl.py to produce outputs/reports/sarl_summary.{json,md}.
#
# Launch with `afterany` (not `afterok`) so we aggregate even if some cells
# failed — the report lists the missing ones.
#
#   ARRAY_JOB_ID=$(sbatch --parsable scripts/slurm/sarl_array.sh)
#   sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/slurm/aggregate.sh

#SBATCH --job-name=sarl-agg
#SBATCH --account=aip-gdumas85
#SBATCH --time=00:30:00
#SBATCH --mem=4000M
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/sarl-agg-%j.out
#SBATCH --error=logs/slurm/sarl-agg-%j.err

# shellcheck source=scripts/slurm/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SCRATCH_OUT="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl"
REPO_OUT="${REPO_ROOT}/outputs/sarl"
mkdir -p "${REPO_OUT}"

echo "[agg] rsync ${SCRATCH_OUT}/ → ${REPO_OUT}/"
if [[ -d "${SCRATCH_OUT}" ]]; then
    rsync -a "${SCRATCH_OUT}/" "${REPO_OUT}/"
else
    echo "[agg] WARN: ${SCRATCH_OUT} does not exist; aggregating whatever is already in ${REPO_OUT}" >&2
fi

# Sprint 07 lock: N=10 seeds (paper-conforming), all 5 games, all 6 settings.
uv run --offline python scripts/aggregate_sarl.py \
    --games "breakout,seaquest,space_invaders,asterix,freeway" \
    --seeds "42,43,44,45,46,47,48,49,50,51" \
    --settings "1,2,3,4,5,6"

echo "[agg] done — see outputs/reports/sarl_summary.md"
