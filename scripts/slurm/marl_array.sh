#!/usr/bin/env bash
# MARL production run — corrected engine (fix a: metacognition acts), Narval.
#
# 480 cells = 4 substrates × 6 settings × 20 seeds × 1M env steps.
#   substrates = {commons_harvest_closed, commons_harvest_partnership, chemistry, territory_inside_out}
#   settings   = {baseline, cascade_1st_no_meta, meta_no_cascade, maps, meta_cascade_2nd, meta_cascade_both}
#   seeds      = 42..61 (20)
#
# Each task runs one cell via scripts/run_marl.py with checkpoint/resume, on the py3.11
# .venv-marl (meltingpot). Idempotent: a present metrics.json => skip; otherwise --resume
# picks up from <out>/checkpoint.pt (saved every save_interval episodes).
#
# Timing measured on Narval A100 (20k-step validation, extrapolated to 1M):
#   fast settings (cascade1=1 : baseline, meta_no_cascade, meta_cascade_2nd)      ~7.6 h / 1M
#   slow settings (cascade1=50: cascade_1st_no_meta, maps, meta_cascade_both)     ~42 h  / 1M
# --time=3-00:00:00 (72h) covers the slowest + margin; --requeue + --resume is the backstop.
# Narval GPU allows up to 7 days. (One --time for the whole array; fast cells just exit early.)
#
# Submission:
#   sbatch scripts/slurm/marl_array.sh                          # full 480 cells
#   sbatch --array=0-3 scripts/slurm/marl_array.sh              # smoke slice (first 4 cells)
#   sbatch --array=0-479%40 scripts/slurm/marl_array.sh         # raise concurrency further (default %20)
#
# Concurrency default %20 (raised from the old %4 lab cap on Rémy's call — the allocation permits
# more). %N is only a self-cap; SLURM still bounds actual concurrency by the def-gdumas85 GPU limit.
# Raise further at submit (e.g. %40) if the queue allows; higher N = faster wall-clock for the 480.

#SBATCH --job-name=marl-prod
#SBATCH --account=def-gdumas85_gpu
#SBATCH --array=0-479%20
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00                # 72h: covers slowest cells (~42h/1M) + margin; resume is backstop
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --requeue                        # survive preemption (resume from checkpoint)
#SBATCH --output=logs/slurm/marl-prod-%A_%a.out
#SBATCH --error=logs/slurm/marl-prod-%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/scripts/run_marl.py" ]]; then
    echo "[marl-prod] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

# ── Modules + env (py3.11 venv-marl; torch bundles CUDA 13, no cuda module needed) ──
module load StdEnv/2023 python/3.11.5

VENV_MARL="${REPO_ROOT}/.venv-marl"
if [[ ! -x "${VENV_MARL}/bin/python" ]]; then
    echo "[marl-prod] ${VENV_MARL} missing — build it per docs/install_marl_drac.md on a login node." >&2
    exit 78
fi
export VIRTUAL_ENV="${VENV_MARL}"
export PATH="${VENV_MARL}/bin:${PATH}"
export PYTHONUNBUFFERED=1
# torch + tensorflow(meltingpot) segfault under multi-threaded OpenMP; force single-threaded.
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export KMP_DUPLICATE_LIB_OK=TRUE

# ── Task → (substrate, setting, seed) ──
SUBSTRATES=(commons_harvest_closed commons_harvest_partnership chemistry territory_inside_out)
SETTINGS=(baseline cascade_1st_no_meta meta_no_cascade maps meta_cascade_2nd meta_cascade_both)
SEEDS=($(seq 42 61))    # 20 seeds

N_SET=${#SETTINGS[@]}    # 6
N_SEED=${#SEEDS[@]}      # 20
N_TASKS=$(( ${#SUBSTRATES[@]} * N_SET * N_SEED ))   # 480

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= N_TASKS )); then
    echo "[marl-prod] TASK_ID=${TASK_ID} >= ${N_TASKS} — out of range" >&2
    exit 64
fi

SUB_IDX=$(( TASK_ID / (N_SET * N_SEED) ))
REM=$(( TASK_ID % (N_SET * N_SEED) ))
SET_IDX=$(( REM / N_SEED ))
SEED_IDX=$(( REM % N_SEED ))

SUBSTRATE=${SUBSTRATES[$SUB_IDX]}
SETTING=${SETTINGS[$SET_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[marl-prod] task=${TASK_ID} substrate=${SUBSTRATE} setting=${SETTING} seed=${SEED} node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

# ── Output dir (rsync'd to outputs/marl/ post-run) ──
OUT_BASE="${MARL_OUT_BASE:-${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl}"
OUT_DIR="${OUT_BASE}/${SUBSTRATE}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"

# Idempotent re-run guard.
if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-prod] ${OUT_DIR}/metrics.json present — skip (already complete)"
    exit 0
fi

python --version
# NUM_ENV_STEPS overridable for validation/timing runs, e.g.
#   sbatch --export=ALL,NUM_ENV_STEPS=30000 --array=360,420 --time=03:00:00 scripts/slurm/marl_array.sh
python scripts/run_marl.py \
    --substrate "${SUBSTRATE}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --num-env-steps "${NUM_ENV_STEPS:-1000000}" \
    --device cuda \
    --output-dir "${OUT_DIR}" \
    --resume

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-prod] FAIL: ${OUT_DIR}/metrics.json missing after run" >&2
    exit 1
fi
echo "[marl-prod] OK cell=${SUBSTRATE}/${SETTING}/${SEED}"
