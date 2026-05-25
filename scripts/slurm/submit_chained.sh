#!/usr/bin/env bash
# Submit a series of sbatch scripts with automatic --dependency=afterany chaining.
#
# Tamia's aip-gdumas85 allocation is shared (Guillaume, Rémy, Nadine, MARL).
# Policy: no more than 2 of my jobs concurrent. Chain instead of parallel.
# This helper enforces that by construction: each submitted job runs only
# after the previous one finishes (afterany → runs even if prev failed;
# lets downstream scripts like aggregate.sh handle partial results).
#
# Usage
# -----
#   scripts/slurm/submit_chained.sh job1.sh [args1...] -- job2.sh [args2...] -- ...
#
# Use a literal `--` as separator between jobs. Args before `--` belong to
# the current job; args after start the next job definition.
#
# Examples
# --------
# Chain bench CPU s1 500k → bench CPU s6 50k → profile CPU s6 25k:
#   scripts/slurm/submit_chained.sh \
#       scripts/slurm/bench_sarl.sh cpu_4c breakout 1 42 500000 -- \
#       scripts/slurm/bench_sarl.sh cpu_4c breakout 6 42 50000 -- \
#       scripts/slurm/profile_sarl.sh breakout 6 42 25000
#
# Prepend sbatch options (e.g. --gpus-per-node=h100:4) by passing them as a
# single quoted string via env var SBATCH_OPTS_<N> for the N-th job (1-indexed):
#   SBATCH_OPTS_2="--gpus-per-node=h100:4 --time=02:00:00" \
#     scripts/slurm/submit_chained.sh \
#       scripts/slurm/bench_sarl.sh cpu_4c breakout 1 42 500000 -- \
#       scripts/slurm/bench_sarl.sh gpu_full breakout 6 42 50000
#
# On exit, prints the chain as a table and leaves it to the user to squeue.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    grep -E "^#([^!]|$)" "${BASH_SOURCE[0]}" | sed 's/^# \?//' | head -40
    exit 64  # EX_USAGE
fi

# ── Split $@ on literal `--` separator into a list of job invocations ──────
declare -a jobs=()
declare -a current=()
for arg in "$@"; do
    if [[ "${arg}" == "--" ]]; then
        if [[ ${#current[@]} -eq 0 ]]; then
            echo "[chain] error: empty job segment (two '--' in a row?)" >&2
            exit 64
        fi
        jobs+=("$(printf '%q ' "${current[@]}")")
        current=()
    else
        current+=("${arg}")
    fi
done
# Last segment (after the final '--', or the only one if no '--' was used).
if [[ ${#current[@]} -gt 0 ]]; then
    jobs+=("$(printf '%q ' "${current[@]}")")
fi

if [[ ${#jobs[@]} -eq 0 ]]; then
    echo "[chain] error: no jobs parsed" >&2
    exit 64
fi

# ── Sanity: warn if > 2 of my jobs are already queued or running ───────────
active=$(squeue --me -h 2>/dev/null | wc -l)
if (( active > 2 )); then
    echo "[chain] WARN: you already have ${active} jobs on the queue. Policy is ≤ 2. Consider cancelling some first." >&2
fi

# ── Submit + chain ─────────────────────────────────────────────────────────
prev_id=""
printf "[chain] %d job(s) to submit\n" "${#jobs[@]}"
for i in "${!jobs[@]}"; do
    idx=$((i + 1))
    opts_var="SBATCH_OPTS_${idx}"
    extra_opts="${!opts_var:-}"

    dep_arg=""
    if [[ -n "${prev_id}" ]]; then
        dep_arg="--dependency=afterany:${prev_id}"
    fi

    # shellcheck disable=SC2086
    cmd="sbatch --parsable ${dep_arg} ${extra_opts} ${jobs[$i]}"
    printf "[chain] [%d/%d] %s\n" "${idx}" "${#jobs[@]}" "${cmd}"

    # Evaluate the command (args are already %q-escaped above).
    job_id=$(eval "${cmd}")
    if [[ -z "${job_id}" ]]; then
        echo "[chain] error: sbatch returned empty job id for segment ${idx}" >&2
        exit 1
    fi
    printf "[chain] [%d/%d] submitted: %s%s\n" "${idx}" "${#jobs[@]}" "${job_id}" "${prev_id:+ (afterany:${prev_id})}"
    prev_id="${job_id}"
done

echo
echo "[chain] done. final job id: ${prev_id}"
echo "[chain] monitor: squeue --me"
