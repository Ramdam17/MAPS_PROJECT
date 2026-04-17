#!/bin/bash
# SLURM batch file for MAPS/MARL (MAPPO + MeltingPot) on Compute Canada
# (Narval, primarily; Rorqual does not expose GPUs for most accounts).
#
# Arguments (passed via --export):
#   SETTING      (1..6)        — MAPS factorial setting, see train_meltingpot.py
#   ENV_CODE     (e.g. TERRITORY_I, HARVEST_P)  — meltingpot.sh case selector
#   SEED         (int)         — random seed
#   N_EPISODES   (int)         — defaults to 2000 if unset
#
# Usage:
#   sbatch --export=SETTING=6,ENV_CODE=TERRITORY_I,SEED=42 train_maps_marl.sbatch
#
# Output goes to $SCRATCH/MAPS_PROJECT/outputs (not $HOME, for quota).

#SBATCH --job-name=maps_marl
#SBATCH --account=rrg-dumasg-ad           # <-- to be confirmed with Rémy / Guillaume
#SBATCH --gres=gpu:a100:1                 # Narval: 4×A100-40GB/node; we take 1
#SBATCH --cpus-per-task=16                # leaves head-room for n_rollout_threads
#SBATCH --mem=64G
#SBATCH --time=23:30:00                   # stay under 24h for default queue
#SBATCH --output=logs/slurm/maps_marl-%x-%j.out
#SBATCH --error=logs/slurm/maps_marl-%x-%j.err

set -euo pipefail

# ───── Required exports ────────────────────────────────────────────────────────
: "${SETTING:?SETTING must be exported: sbatch --export=SETTING=N,...}"
: "${ENV_CODE:?ENV_CODE must be exported (e.g. TERRITORY_I, HARVEST_P)}"
: "${SEED:?SEED must be exported}"
N_EPISODES="${N_EPISODES:-2000}"

# ───── Modules (Narval StdEnv/2023) ────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11 cuda/12.2 gcc/12.3

# SubprocVecEnv spawns N Python workers; each inherits OMP/MKL thread pools.
# Setting these to 1 prevents (threads × workers) oversubscription of CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ───── Workspace layout ────────────────────────────────────────────────────────
# $SLURM_TMPDIR: node-local NVMe, fast I/O, wiped at job end.
# $SCRATCH: persistent, slower — we use it for outputs/checkpoints.
PROJECT_HOME="${PROJECT_HOME:-${HOME}/projects/def-dumasg/${USER}/MAPS_PROJECT}"
OUTPUT_ROOT="${SCRATCH}/MAPS_PROJECT/outputs"
LOG_ROOT="${SCRATCH}/MAPS_PROJECT/logs/slurm"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

echo "[sbatch] Staging project to ${SLURM_TMPDIR} ..."
rsync -a --exclude='.git' --exclude='outputs' --exclude='logs' \
      --exclude='.venv' --exclude='__pycache__' \
      "${PROJECT_HOME}/" "${SLURM_TMPDIR}/MAPS_PROJECT/"

cd "${SLURM_TMPDIR}/MAPS_PROJECT/MARL"

# ───── Environment ─────────────────────────────────────────────────────────────
# Assumes a pre-built virtualenv at $PROJECT_HOME/.venv (built once via:
#   module load python/3.11 && virtualenv .venv && pip install -e .[marl,dev])
source "${PROJECT_HOME}/.venv/bin/activate"

echo "[sbatch] python: $(which python)"
python --version
python -c "import torch; print(f'[sbatch] torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')"

# ───── Ensure n_rollout_threads matches CPU allocation ─────────────────────────
# NOTE: this is a *purely computational* knob (see docs/reproduction/deviations.md).
# It changes the effective batch seen by PPO but does not alter model behaviour.
N_ROLLOUT="${N_ROLLOUT:-$((SLURM_CPUS_PER_TASK - 2))}"
export N_ROLLOUT

echo "[sbatch] setting=${SETTING} env=${ENV_CODE} seed=${SEED} rollout_threads=${N_ROLLOUT}"

# ───── Run ─────────────────────────────────────────────────────────────────────
# meltingpot.sh args positionnels:
#   $1 environment  $2 seed  $3 module  $4 hidden  $5 units  $6 topk
#   $7 optimizer    $8 run   $9 rollout $10 setting
./meltingpot.sh \
    "${ENV_CODE}" \
    "${SEED}" \
    LSTM \
    100 \
    1 \
    1 \
    ADAM \
    "${SLURM_JOB_ID}" \
    "${N_ROLLOUT}" \
    "${SETTING}"

# ───── Persist outputs ─────────────────────────────────────────────────────────
RUN_OUT="${OUTPUT_ROOT}/setting${SETTING}/${ENV_CODE}/seed${SEED}-job${SLURM_JOB_ID}"
mkdir -p "${RUN_OUT}"

rsync -a \
  "${SLURM_TMPDIR}/MAPS_PROJECT/MARL/MAPPO-ATTENTIOAN/onpolicy/scripts/results/" \
  "${RUN_OUT}/results/" || true

rsync -a \
  "${SLURM_TMPDIR}/MAPS_PROJECT/MARL/MAPPO-ATTENTIOAN/logs/" \
  "${RUN_OUT}/logs/" || true

echo "[sbatch] DONE. Results synced to ${RUN_OUT}"
