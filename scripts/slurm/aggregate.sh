#!/usr/bin/env bash
# Sprint 07 — post-array aggregation on Narval.
#
# Submit AFTER the sarl_array.sh job has finished (or use --dependency=afterok):
#   sbatch --account=<rrg-group> \
#          --dependency=afterok:<ARRAY_JOB_ID> \
#          scripts/slurm/aggregate.sh

#SBATCH --job-name=maps-sarl-agg
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/maps-sarl-agg-%j.out
#SBATCH --error=logs/slurm/maps-sarl-agg-%j.err

set -euo pipefail

module load python/3.12
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

# Copy outputs from $SCRATCH to the repo working tree so aggregate_sarl.py
# finds them under outputs/sarl/.
SCRATCH_OUT="${SCRATCH:-$PWD}/maps/outputs/sarl"
REPO_OUT="$PWD/outputs/sarl"
mkdir -p "$REPO_OUT"
rsync -a --delete "$SCRATCH_OUT/" "$REPO_OUT/"

python scripts/aggregate_sarl.py \
    --games "breakout,seaquest,space_invaders,asterix,freeway" \
    --seeds "42,43,44,45,46" \
    --settings "1,2,3,4,5,6"

echo "Aggregation done — see outputs/reports/sarl_summary.md"
