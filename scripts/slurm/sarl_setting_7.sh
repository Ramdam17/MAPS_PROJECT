#!/usr/bin/env bash
# Paper Setting 7 (ACB = Actor-Critic Baseline, Young & Tian 2019, λ=0.8)
# for the 5 MinAtar games × 3 seeds = 15 cells.
#
# Closes deviation D-sarl-setting-7 in docs/reproduction/deviations.md (§B.7).
# Plan reference : docs/plans/plan-20260519-acb-setting-7.md (Sub-plan A).
#
# Algorithmic isolation : ACB is structurally different from settings 1-6
# (no replay buffer, online actor-critic with eligibility traces, custom
# debiased RMSprop). It does NOT touch the DQN code path ; running this job
# concurrently with the legacy SARL settings 1-6 arrays is safe.
#
# Per-cell wall : measured 0.002 s/frame on tc11101 (CPU). 500 000 frames
# × 3 seeds = ~17 min × 3 = ~51 min per task. 5 tasks at %3 concurrency
# = ~1h30 total wall time.
#
# Usage :
#   sbatch scripts/slurm/sarl_setting_7.sh
#
# Outputs : $SCRATCH/maps/outputs/sarl/<game>/setting-7/seed-<N>/{summary.json,...}.

#SBATCH --job-name=sarl-s7-acb
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-4%2                    # 5 tasks (one per game), 2 concurrent
#SBATCH --time=03:00:00                  # ~51 min observed, 3.5× safety margin
# %2 is the sustainable default for the aip-gdumas85 shared queue : together
# with the 2 concurrent BS / AGL tasks of the Phase γ Know-Thyself arrays
# (~4 h overlap window), this caps total concurrency at 4.
#SBATCH --mem=4096M
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/sarl-s7-%A_%a.out
#SBATCH --error=logs/slurm/sarl-s7-%A_%a.err

set -euo pipefail

GAMES=(space_invaders breakout seaquest asterix freeway)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
GAME=${GAMES[$TASK_ID]}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[sarl-s7] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12

VENV="${REPO_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[sarl-s7] ${VENV} missing — run 'uv sync --extra sarl --extra dev' first." >&2
    exit 78
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1

# Seeds : 3 per game, matches paper Table 6 (D-sarl-seeds resolved as 3).
SEEDS=(42 43 44)

echo "[sarl-s7] task=${TASK_ID} game=${GAME} seeds=${SEEDS[*]}"

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl/${GAME}/setting-7"
mkdir -p "${OUT_BASE}"

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_BASE}/seed-${SEED}"
    if [[ -s "${OUT_DIR}/summary.json" ]]; then
        echo "[sarl-s7] skip game=${GAME} seed=${SEED} (summary already present)"
        continue
    fi
    uv run --offline python scripts/run_sarl.py \
        --game "${GAME}" \
        --setting 7 \
        --seed "${SEED}"
done

# Completeness check.
N_DONE=$(find "${OUT_BASE}" -name summary.json 2>/dev/null | wc -l)
echo "[sarl-s7] task=${TASK_ID} game=${GAME} done=${N_DONE}/${#SEEDS[@]}"
if (( N_DONE < ${#SEEDS[@]} )); then
    echo "[sarl-s7] FAIL: incomplete — only ${N_DONE}/${#SEEDS[@]} cells written" >&2
    exit 1
fi
echo "[sarl-s7] OK game=${GAME}"
