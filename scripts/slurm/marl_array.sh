#!/usr/bin/env bash
# Sprint 09 — MARL reproduction array on DRAC Tamia.
#
# 72 cells = 4 substrates × 6 settings × 3 seeds (paper §A.4 + factorial_marl.yaml).
#   substrates = {commons_harvest_closed, commons_harvest_partnership,
#                 chemistry, territory_inside_out}
#   settings   = {baseline, cascade_1st_no_meta, meta_no_cascade, maps,
#                 meta_cascade_2nd, meta_cascade_both}
#   seeds      = {42, 43, 44}
#
# Wall-time budget : paper §B.4 reports ~16 h / seed on A100 for 15 M env
# steps. Tamia H100 / H200 is faster → 20 h cap with 4 h buffer = 24 h wall.
# If setting 4-6 (meta + cascade_1st=50) bust wall, split into a 2nd array
# with shorter num_env_steps per cell (see D-marl-scope notes).
#
# Concurrency : aip-gdumas85 is lab-shared (Guillaume + Rémy + MARL). Cap
# at %3 simultaneous jobs per saved feedback (feedback_shared_queue_dependencies.md).
#
# Submission :
#   sbatch scripts/slurm/marl_array.sh                              # full 72 cells, GPU
#   sbatch --array=0-11 scripts/slurm/marl_array.sh                 # smoke slice
#   sbatch --export=DEVICE=cpu scripts/slurm/marl_array.sh          # CPU (dev only)
#
# Outputs land in $SCRATCH/maps/outputs/marl/<substrate>/setting-<id>/seed-<SEED>/
# and are rsync'd to outputs/marl/ post-run (TBD — plan E.18).

#SBATCH --job-name=marl-array
#SBATCH --account=aip-gdumas85
#SBATCH --array=0-71%3                  # 72 cells ; shared queue cap ≤ 3 concurrent
#SBATCH --time=24:00:00                 # 20 h paper budget + 4 h buffer
#SBATCH --mem=16384M                    # per-cell peak ≈ 8-12 GB (vision CNN + N policies)
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1               # H100 default ; override to h200:1 if available
#SBATCH --requeue                       # survive pre-emption (checkpoint WIP — see below)
#SBATCH --output=logs/slurm/marl-array-%A_%a.out
#SBATCH --error=logs/slurm/marl-array-%A_%a.err

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ ⚠  MARL checkpoint resume is NOT implemented yet (E.13+).               │
# │                                                                         │
# │ A requeue after pre-emption will restart the cell from episode 0. For   │
# │ short smoke runs that's fine ; for the full 15 M-step run we need       │
# │ checkpoint save/load to survive. Track this in docs/TODO.md under       │
# │ "MARL : resume from checkpoint" before launching the full array.        │
# └─────────────────────────────────────────────────────────────────────────┘

set -euo pipefail

# Device : cuda (default) or cpu. Override via `--export=DEVICE=cpu`.
DEVICE=${DEVICE:-cuda}

# ── Repo root + logging setup ─────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[marl-array] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

# ── Modules + env ────────────────────────────────────────────────────────
# dmlab2d / meltingpot only install on Python 3.11 → use dedicated .venv-marl.
module load StdEnv/2023 python/3.11.5 cuda/12.6

VENV_MARL="${REPO_ROOT}/.venv-marl"
if [[ ! -x "${VENV_MARL}/bin/python" ]]; then
    echo "[marl-array] ${VENV_MARL} missing — run install per docs/install_marl_drac.md on login node first." >&2
    exit 78
fi

export VIRTUAL_ENV="${VENV_MARL}"
export PATH="${VENV_MARL}/bin:${PATH}"
export PYTHONUNBUFFERED=1
# HyPyP / some deps require these on compute nodes with no internet.
export TOKENIZERS_PARALLELISM=false

# ── Task → (substrate, setting, seed) decode ────────────────────────────
SUBSTRATES=(commons_harvest_closed commons_harvest_partnership chemistry territory_inside_out)
SETTINGS=(baseline cascade_1st_no_meta meta_no_cascade maps meta_cascade_2nd meta_cascade_both)
SEEDS=(42 43 44)

N_SUBSTRATES=${#SUBSTRATES[@]}
N_SETTINGS=${#SETTINGS[@]}
N_SEEDS=${#SEEDS[@]}
EXPECTED_TASKS=$(( N_SUBSTRATES * N_SETTINGS * N_SEEDS ))  # 72

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= EXPECTED_TASKS )); then
    echo "[marl-array] TASK_ID=${TASK_ID} >= ${EXPECTED_TASKS} — out of range" >&2
    exit 64
fi

SUB_IDX=$(( TASK_ID / (N_SETTINGS * N_SEEDS) ))
REM=$(( TASK_ID % (N_SETTINGS * N_SEEDS) ))
SET_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

SUBSTRATE=${SUBSTRATES[$SUB_IDX]}
SETTING=${SETTINGS[$SET_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[marl-array] task=${TASK_ID} substrate=${SUBSTRATE} setting=${SETTING} seed=${SEED} device=${DEVICE}"

# ── GPU sanity ───────────────────────────────────────────────────────────
if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[marl-array] DEVICE=cuda but no GPU visible — submit with --gres=gpu:h100:1" >&2
        exit 64
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
fi

# ── Output dir ───────────────────────────────────────────────────────────
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl"
OUT_DIR="${OUT_BASE}/${SUBSTRATE}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

# Idempotent re-run guard.
if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-array] ${OUT_DIR}/metrics.json already present — skip"
    exit 0
fi

# ── Run ──────────────────────────────────────────────────────────────────
python --version

python scripts/run_marl.py \
    --substrate "${SUBSTRATE}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --output-dir "${OUT_DIR}"

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-array] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[marl-array] OK cell=${SUBSTRATE}/${SETTING}/${SEED}"
