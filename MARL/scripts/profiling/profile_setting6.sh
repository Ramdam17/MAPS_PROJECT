#!/bin/bash
# Short cProfile run for MAPPO+MeltingPot setting 6 — WORST CASE:
# meta=True + cascade_iterations1=50 + cascade_iterations2=50.
# This is the setting that should bear the brunt of the speedup effort.
#
# Outputs:
#   outputs/profiling/setting6_<date>.prof
#   outputs/profiling/setting6_<date>.log
#
# Use the same analysis workflow as profile_setting1.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve python ($PY) and verify torch + meltingpot + dmlab2d import.
# Fails fast with a helpful message on macOS (MARL deps are Linux-only).
# shellcheck source=_preflight.sh
source "${SCRIPT_DIR}/_preflight.sh"

MARL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_PY="${MARL_DIR}/MAPPO-ATTENTIOAN/onpolicy/scripts/train/train_meltingpot.py"
OUT_DIR="${MARL_DIR}/outputs/profiling"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${MARL_DIR}/MAPPO-ATTENTIOAN"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUBSTRATE="territory__inside_out"
AGENTS=5
SEED=42

PROF="${OUT_DIR}/setting6_${STAMP}.prof"
LOG="${OUT_DIR}/setting6_${STAMP}.log"

echo "[profile] setting 6 (meta=True, cascade1=50, cascade2=50) on ${SUBSTRATE}"
echo "[profile] stats -> ${PROF}"
echo "[profile] log   -> ${LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ${PY} "${TRAIN_PY}" \
    --setting 6 \
    --profile \
    --profile-out "${PROF}" \
    --load_model False \
    --model_dir None \
    --run_num 0 \
    --optimizer ADAM \
    --use_valuenorm False \
    --use_popart True \
    --env_name "Meltingpot" \
    --experiment_name "profile_setting_6" \
    --substrate_name "${SUBSTRATE}" \
    --num_agents "${AGENTS}" \
    --seed "${SEED}" \
    --n_rollout_threads 1 \
    --use_wandb False \
    --share_policy False \
    --use_centralized_V False \
    --use_attention False \
    --use_naive_recurrent_policy True \
    --use_recurrent_policy True \
    --hidden_size 100 \
    --use_gae True \
    --episode_length 1000 \
    --attention_module LSTM \
    --algorithm_name mappo \
    --num_episodes 3 \
    --num_env_steps 30000 2>&1 | tee "${LOG}"

echo ""
echo "[profile] DONE. Top 30 by cumulative time:"
${PY} -c "import pstats,sys; p=pstats.Stats('${PROF}'); p.sort_stats('cumulative').print_stats(30)" | tee -a "${LOG}"

echo ""
echo "[profile] Top 30 by total time (self):"
${PY} -c "import pstats,sys; p=pstats.Stats('${PROF}'); p.sort_stats('tottime').print_stats(30)" | tee -a "${LOG}"
