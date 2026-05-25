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
# Wall-time budget : num_env_steps = 300k (paper §4 p.15). Bench E.17b
# measured 1 M steps = 5h22 on baseline H100 → 300k ≈ 1h35 baseline, up to
# ~4-6 h for cascade×50 settings. 24 h wall leaves ample margin ; plus
# checkpoint / resume (E.17a) survives any requeue.
#
# Concurrency : 4 parallel tasks is the lab-negotiated cap
# (feedback_shared_queue_dependencies.md).
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
#SBATCH --array=0-71%4                  # 72 cells ; shared queue cap = 4 concurrent
#SBATCH --time=24:00:00                 # 20 h paper budget + 4 h buffer
#SBATCH --mem=16384M                    # per-cell peak ≈ 8-12 GB (vision CNN + N policies)
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4          # Tamia H100 = whole-node allocation ; we scope to GPU 0 in-script
#SBATCH --requeue                       # survive pre-emption (checkpoint WIP — see below)
#SBATCH --output=logs/slurm/marl-array-%A_%a.out
#SBATCH --error=logs/slurm/marl-array-%A_%a.err

# Checkpoint save/load (E.17a) is wired : the CLI's --resume flag is a
# silent no-op if no checkpoint exists, so a requeued task picks up from
# the last save_interval automatically. No action needed at the script level.

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

# Tamia H100 allocates whole-node (4 GPUs). Scope each cell to GPU 0 so
# in-run `torch.cuda.current_device()` stays deterministic.
export CUDA_VISIBLE_DEVICES=0

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
        echo "[marl-array] DEVICE=cuda but no GPU visible — submit with --gpus-per-node=h100:4" >&2
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
    --output-dir "${OUT_DIR}" \
    --resume

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-array] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[marl-array] OK cell=${SUBSTRATE}/${SETTING}/${SEED}"
