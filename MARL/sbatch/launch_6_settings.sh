#!/bin/bash
# Driver script for the 6-setting factorial reproduction (Vargas et al. MAPS, TMLR).
#
# Submits one sbatch job per (setting, seed, env) combination. Uses
# train_maps_marl.sbatch as the unit of work.
#
# Usage:
#   ./launch_6_settings.sh [ENV_CODE] [N_SEEDS]
#
# Examples:
#   ./launch_6_settings.sh TERRITORY_I 3      # 6 × 3 = 18 jobs
#   ./launch_6_settings.sh HARVEST_P   5      # 6 × 5 = 30 jobs
#
# After all jobs finish, aggregate results with:
#   scripts/analysis/aggregate_marl_results.py  (to be written in Phase >=4)

set -euo pipefail

ENV_CODE="${1:-TERRITORY_I}"
N_SEEDS="${2:-3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/train_maps_marl.sbatch"

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "ERROR: ${SBATCH_FILE} not found"; exit 1
fi

# Seeds: deterministic family — 42, 43, 44, ... — keeps reproduction aligned
# with the seeding convention in src/maps/utils/seeding.py.
BASE_SEED=42

echo "[launch] env=${ENV_CODE} seeds=${N_SEEDS} (base=${BASE_SEED})  settings=1..6"
echo "[launch] sbatch file: ${SBATCH_FILE}"
echo ""

for SETTING in 1 2 3 4 5 6; do
    for ((i=0; i<N_SEEDS; i++)); do
        SEED=$((BASE_SEED + i))
        JOB_NAME="maps_s${SETTING}_${ENV_CODE}_seed${SEED}"
        echo "  sbatch --job-name=${JOB_NAME} --export=SETTING=${SETTING},ENV_CODE=${ENV_CODE},SEED=${SEED}"
        sbatch \
            --job-name="${JOB_NAME}" \
            --export=ALL,SETTING="${SETTING}",ENV_CODE="${ENV_CODE}",SEED="${SEED}" \
            "${SBATCH_FILE}"
    done
done

echo ""
echo "[launch] Submitted $((6 * N_SEEDS)) jobs. Monitor with: squeue -u \$USER"
