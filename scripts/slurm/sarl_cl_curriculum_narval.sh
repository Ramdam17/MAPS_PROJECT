#!/usr/bin/env bash
# Phase F.4 — SARL+CL curriculum reproduction on DRAC Narval (A100 GPU).
#
# Reproduces Figure 7 of the MAPS paper (Vargas et al., TMLR submission, p.17) :
# sequential training across Breakout → Space Invaders → Seaquest → Freeway
# with the 3-term CL loss (task + weight-reg + feature).
#
# Matrix : 18 chains = 6 settings × 3 seeds.
# Each chain : 4 sequential stages with same (setting, seed), curriculum order
# read from config/training/sarl_cl.yaml (cl.curriculum_order).
# Total : 72 trainings of 100k frames each (paper text p.17 explicit).
#
# Loss weights : (0.4, 0.4, 0.2) per paper text p.17 "optimal weights"
# (Figure 7 reproduction). See deviations.md D-cl-weights (2026-05-19 reversal).
#
# Stage 0 (Breakout) : no teacher, --adaptive only (AdaptiveQNetwork backbone
#                       so later stages can load partial state dict).
# Stages 1-3         : --adaptive --curriculum --teacher-load-path=<prev/checkpoint.pt>
#
# Output layout : $SCRATCH/maps/outputs/sarl_cl_curriculum/setting-<S>/seed-<Z>/stage-<i>-<game>/
# Idempotent : per-stage metrics.json guard + per-call --resume (pre-emption safe).
#
# Usage :
#   sbatch --array=0%1 scripts/slurm/sarl_cl_curriculum_narval.sh   # smoke (1 chain)
#   sbatch scripts/slurm/sarl_cl_curriculum_narval.sh               # full 18 chains

#SBATCH --job-name=sarl-cl-curriculum
#SBATCH --account=def-gdumas85_gpu
#SBATCH --array=0-17%18
#SBATCH --time=08:00:00
#SBATCH --mem=8192M
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/sarl-cl-curriculum-%A_%a.out
#SBATCH --error=logs/slurm/sarl-cl-curriculum-%A_%a.err

set -euo pipefail

DEVICE=${DEVICE:-cuda}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[curriculum] REPO_ROOT=${REPO_ROOT} does not look like a MAPS checkout" >&2
    exit 78
fi

module load StdEnv/2023 python/3.12 cuda/12.2

VENV="${REPO_ROOT}/.venv-narval"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[curriculum] ${VENV} missing — run venv setup on the login node first." >&2
    echo "[curriculum] Expected layout: \$SCRATCH/maps/venv-narval + symlink .venv-narval" >&2
    exit 78
fi

export PYTHONUNBUFFERED=1

# ── Read curriculum order from YAML (no hardcode) ──────────────────────────
CL_YAML="${REPO_ROOT}/config/training/sarl_cl.yaml"
mapfile -t STAGES < <("${VENV}/bin/python" -c "
import yaml, sys
with open('${CL_YAML}') as f:
    cfg = yaml.safe_load(f)
for game in cfg['cl']['curriculum_order']:
    print(game)
")

if [[ ${#STAGES[@]} -ne 4 ]]; then
    echo "[curriculum] cl.curriculum_order in ${CL_YAML} must have exactly 4 games (got ${#STAGES[@]}: ${STAGES[*]})" >&2
    exit 78
fi

# ── Chain decode : TASK_ID ∈ [0,18) → (setting ∈ [1,6], seed ∈ {42,43,44}) ──
SEEDS=(42 43 44)
N_SETTINGS=6
N_SEEDS=${#SEEDS[@]}
EXPECTED_CHAINS=$(( N_SETTINGS * N_SEEDS ))  # 18

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= EXPECTED_CHAINS )); then
    echo "[curriculum] TASK_ID=${TASK_ID} out of range (max ${EXPECTED_CHAINS})" >&2
    exit 64
fi

SETTING=$(( TASK_ID / N_SEEDS + 1 ))
SEED=${SEEDS[$(( TASK_ID % N_SEEDS ))]}

echo "[curriculum] chain=${TASK_ID} setting=${SETTING} seed=${SEED} stages=(${STAGES[*]}) device=${DEVICE}"

# ── GPU sanity ─────────────────────────────────────────────────────────────
if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[curriculum] no GPU visible — submit with --gres=gpu:a100:1" >&2
        exit 64
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
fi

# ── Chain output layout ────────────────────────────────────────────────────
OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/sarl_cl_curriculum"
CHAIN_DIR="${OUT_BASE}/setting-${SETTING}/seed-${SEED}"
mkdir -p "${CHAIN_DIR}"

# ── Run the 4 stages sequentially, passing checkpoint forward ──────────────
PREV_CKPT=""
for i in "${!STAGES[@]}"; do
    GAME="${STAGES[$i]}"
    STAGE_DIR="${CHAIN_DIR}/stage-${i}-${GAME}"
    mkdir -p "${STAGE_DIR}"

    # Idempotent guard : if this stage already produced metrics.json, skip.
    # Useful after pre-emption requeue and for re-running an incomplete chain.
    if [[ -s "${STAGE_DIR}/metrics.json" ]]; then
        echo "[curriculum] stage=${i} game=${GAME} already complete — skip"
        if [[ ! -s "${STAGE_DIR}/checkpoint.pt" ]]; then
            echo "[curriculum] WARNING: ${STAGE_DIR}/metrics.json exists but checkpoint.pt missing" >&2
            echo "[curriculum] subsequent stages will fail without a teacher checkpoint" >&2
        fi
        PREV_CKPT="${STAGE_DIR}/checkpoint.pt"
        continue
    fi

    echo "[curriculum] === stage=${i} game=${GAME} setting=${SETTING} seed=${SEED} ==="

    if [[ $i -eq 0 ]]; then
        # First stage : no teacher network (no prior task), but --adaptive so
        # the AdaptiveQNetwork backbone matches the variable in_channels of
        # subsequent stages (paper p.9 : "max number of channels across all
        # environments" = 10 for Seaquest).
        "${VENV}/bin/python" scripts/run_sarl_cl.py \
            --game "${GAME}" \
            --setting "${SETTING}" \
            --seed "${SEED}" \
            --adaptive \
            --output-dir "${STAGE_DIR}" \
            --resume \
            -o "device=${DEVICE}"
    else
        if [[ ! -s "${PREV_CKPT}" ]]; then
            echo "[curriculum] FAIL: teacher checkpoint ${PREV_CKPT} missing" >&2
            exit 1
        fi
        # Stages 1-3 : full CL — adaptive backbone + 3-term loss + teacher.
        "${VENV}/bin/python" scripts/run_sarl_cl.py \
            --game "${GAME}" \
            --setting "${SETTING}" \
            --seed "${SEED}" \
            --adaptive \
            --curriculum \
            --teacher-load-path "${PREV_CKPT}" \
            --output-dir "${STAGE_DIR}" \
            --resume \
            -o "device=${DEVICE}"
    fi

    if [[ ! -s "${STAGE_DIR}/metrics.json" ]]; then
        echo "[curriculum] FAIL: ${STAGE_DIR}/metrics.json missing or empty after run" >&2
        exit 1
    fi
    if [[ ! -s "${STAGE_DIR}/checkpoint.pt" ]]; then
        echo "[curriculum] FAIL: ${STAGE_DIR}/checkpoint.pt missing or empty after run" >&2
        exit 1
    fi

    PREV_CKPT="${STAGE_DIR}/checkpoint.pt"
    echo "[curriculum] OK stage=${i} game=${GAME} → checkpoint=${PREV_CKPT}"
done

echo "[curriculum] CHAIN COMPLETE setting=${SETTING} seed=${SEED} chain=${TASK_ID}"
