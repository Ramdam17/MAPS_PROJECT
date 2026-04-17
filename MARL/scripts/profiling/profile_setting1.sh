#!/bin/bash
# Short cProfile run for MAPPO+MeltingPot setting 1 (no meta, cascade=1).
# Purpose: baseline profile to compare against setting 6 and spot overhead
# that is NOT caused by the cascade/meta components.
#
# Outputs:
#   outputs/profiling/setting1_<date>.prof   (cProfile binary)
#   outputs/profiling/setting1_<date>.log    (stdout + timings breakdown)
#
# Analyse with:
#   snakeviz outputs/profiling/setting1_<date>.prof
#   gprof2dot -f pstats <file> | dot -Tsvg > setting1.svg

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MARL_DIR="${REPO_ROOT}"
TRAIN_PY="${MARL_DIR}/MAPPO-ATTENTIOAN/onpolicy/scripts/train/train_meltingpot.py"
OUT_DIR="${REPO_ROOT}/outputs/profiling"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${MARL_DIR}/MAPPO-ATTENTIOAN"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUBSTRATE="territory__inside_out"   # small map, 5 agents — fast to profile
AGENTS=5
SEED=42

PROF="${OUT_DIR}/setting1_${STAMP}.prof"
LOG="${OUT_DIR}/setting1_${STAMP}.log"

echo "[profile] setting 1 (no meta, cascade=1) on ${SUBSTRATE}"
echo "[profile] stats -> ${PROF}"
echo "[profile] log   -> ${LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${TRAIN_PY}" \
    --setting 1 \
    --profile \
    --profile-out "${PROF}" \
    --load_model False \
    --model_dir None \
    --run_num 0 \
    --optimizer ADAM \
    --use_valuenorm False \
    --use_popart True \
    --env_name "Meltingpot" \
    --experiment_name "profile_setting_1" \
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
    --num_episodes 5 \
    --num_env_steps 50000 2>&1 | tee "${LOG}"

echo ""
echo "[profile] DONE. Top 30 by cumulative time:"
python -c "import pstats,sys; p=pstats.Stats('${PROF}'); p.sort_stats('cumulative').print_stats(30)" | tee -a "${LOG}"
