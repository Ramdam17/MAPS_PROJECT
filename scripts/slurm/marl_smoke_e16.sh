#!/usr/bin/env bash
# E.16 — MARL smoke run on DRAC Tamia H100.
#
# Single cell : commons_harvest_closed × baseline × seed 42 × 100 k env steps.
# Validates the full pipeline end-to-end before the 72-cell array :
# - .venv-marl (Python 3.11 + dmlab2d + meltingpot) loads
# - MeltingPotEnv + DownSamplingSubstrateWrapper builds on GPU
# - MeltingpotRunner runs N episodes, trains 6 agents, writes metrics.json
# - Wall-clock budget : ≤ 1 h on H100
#
# Usage :
#   sbatch scripts/slurm/marl_smoke_e16.sh                 # defaults
#   sbatch scripts/slurm/marl_smoke_e16.sh chemistry maps  # override cell
#
# Positional args (in order) : substrate, setting, seed, num_env_steps.
# Outputs land in $SCRATCH/maps/outputs/marl/<substrate>/setting-<id>/seed-<N>/metrics.json.

#SBATCH --job-name=marl-smoke
#SBATCH --account=aip-gdumas85
#SBATCH --time=01:00:00
#SBATCH --mem=16384M
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=logs/slurm/marl-smoke-%j.out
#SBATCH --error=logs/slurm/marl-smoke-%j.err

# Tamia : H100 nodes allocate whole-node (4 GPUs). We only use 1, but reserve
# the pack. Set CUDA_VISIBLE_DEVICES=0 inside the script to scope usage.

set -euo pipefail

# Positional args — simple defaults for the "standard smoke" cell.
SUBSTRATE=${1:-commons_harvest_closed}
SETTING=${2:-baseline}
SEED=${3:-42}
NUM_ENV_STEPS=${4:-100000}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[marl-smoke] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.11.5 cuda/12.6

VENV_MARL="${REPO_ROOT}/.venv-marl"
if [[ ! -x "${VENV_MARL}/bin/python" ]]; then
    echo "[marl-smoke] ${VENV_MARL} missing — run install per docs/install_marl_drac.md first." >&2
    exit 78
fi

export VIRTUAL_ENV="${VENV_MARL}"
export PATH="${VENV_MARL}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# H100 is allocated whole-node (4 GPUs) on Tamia ; scope our script to GPU 0.
export CUDA_VISIBLE_DEVICES=0

echo "[marl-smoke] substrate=${SUBSTRATE} setting=${SETTING} seed=${SEED} num_env_steps=${NUM_ENV_STEPS}"
echo "[marl-smoke] python=$(python --version 2>&1)"

# GPU visibility sanity (fatal if absent — we requested --gres=gpu:h100:1).
if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[marl-smoke] no GPU visible — did the allocation include --gres=gpu:h100:1 ?" >&2
    exit 64
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

# Output dir — $SCRATCH on DRAC, project tree otherwise.
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl"
OUT_DIR="${OUT_BASE}/${SUBSTRATE}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${OUT_DIR}"
echo "[marl-smoke] out_dir=${OUT_DIR}"

# ── Run ──────────────────────────────────────────────────────────────────
# Idempotent re-run guard : skip if metrics.json already present.
if [[ -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-smoke] ${OUT_DIR}/metrics.json already present — skip"
    exit 0
fi

time python scripts/run_marl.py \
    --substrate "${SUBSTRATE}" \
    --setting "${SETTING}" \
    --seed "${SEED}" \
    --device cuda \
    --num-env-steps "${NUM_ENV_STEPS}" \
    --output-dir "${OUT_DIR}"

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
    echo "[marl-smoke] FAIL: ${OUT_DIR}/metrics.json missing or empty" >&2
    exit 1
fi
echo "[marl-smoke] OK — metrics persisted ($(stat -c %s "${OUT_DIR}/metrics.json") bytes)"
