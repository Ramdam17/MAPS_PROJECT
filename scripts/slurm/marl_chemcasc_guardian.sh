#!/usr/bin/env bash
# Chemistry-cascade guardian — keeps the 60 slow cells (chemistry x cascade_1st_no_meta /
# maps / meta_cascade_both, all 20 seeds) moving, valid for every seed automatically.
#
# WHY: chemistry (8 agents) x cascade1=50 needs ~106h/1M — longer than any reasonably
# schedulable single request on this cluster. A single 6-day array (job 65650461, 2026-07-15)
# sat in PENDING for 3+ days with ZERO cells started: SLURM has no user-facing priority boost
# (only QOS=normal available; nice cannot be lowered — Access/permission denied), and a 6-day
# reservation is hard for backfill regardless of queue depth. The practical lever a normal user
# DOES have is REQUEST LENGTH: shorter asks schedule far more easily.
#
# APPROACH: request --time=3-00:00:00 (72h, same as the main array — which IS running) per
# cell instead of one 6-day block. Since 72h < ~106h needed, a chemistry-cascade cell will
# likely need 2 submissions to finish (checkpoint/resume already lands every save_interval
# episodes — see scripts/run_marl.py). This guardian is what makes that automatic: each tick
# it checks all 60 (substrate=chemistry, setting in {cascade_1st_no_meta,maps,meta_cascade_both},
# seed in 42..61) cells and, for any that are NOT done and NOT currently PENDING/RUNNING
# (i.e. it finished a 72h slice via TIMEOUT, or failed), resubmits JUST that cell immediately
# — "priority" here means "relaunched the moment it stops," since actual SLURM priority
# cannot be manually raised.
#
# Same daily-internal-loop + reschedule-at-start pattern as scripts/slurm/marl_watch.sh v2
# (a v1 per-hour self-resubmit chain died once on a transient node I/O error before reaching
# its reschedule line — this shape survives that class of failure).
#
# Start:  sbatch scripts/slurm/marl_chemcasc_guardian.sh
# Stop:   scancel --name=marl-chemcasc-g --name=marl-chemcasc
# Read:   tail -f $SCRATCH/marl_reports/chemcasc_guardian.log

#SBATCH --job-name=marl-chemcasc-g
#SBATCH --account=def-gdumas85_cpu
#SBATCH --time=1-01:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/marl-chemcasc-g-%j.out
#SBATCH --error=logs/slurm/marl-chemcasc-g-%j.out

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"
SELF="${REPO_ROOT}/scripts/slurm/marl_array.sh"
GUARDIAN_SELF="${REPO_ROOT}/scripts/slurm/marl_chemcasc_guardian.sh"
OB="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl"
REP="${SCRATCH:-${REPO_ROOT}/outputs}/marl_reports"
mkdir -p "${REP}"
LOG="${REP}/chemcasc_guardian.log"

SUBSTRATES=(commons_harvest_closed commons_harvest_partnership chemistry territory_inside_out)
SETTINGS=(baseline cascade_1st_no_meta meta_no_cascade maps meta_cascade_2nd meta_cascade_both)
CHEM_IDX=2       # chemistry
CASCADE_OFFS=(1 3 5)   # cascade_1st_no_meta, maps, meta_cascade_both (within a 24-cell seed block)
N_SEEDS=20

tick() {
    local TS n_done=0 n_flight=0 n_sub=0 s off tid sub_i set_i seed cell out_dir state
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    for s in $(seq 0 $((N_SEEDS - 1))); do
        for off in "${CASCADE_OFFS[@]}"; do
            tid=$(( s * 24 + CHEM_IDX * 6 + off ))
            seed=$((42 + s))
            set_i=${off}
            cell="chemistry/setting-${SETTINGS[$set_i]}/seed-${seed}"
            out_dir="${OB}/${cell}"

            if [[ -s "${out_dir}/metrics.json" ]]; then
                n_done=$((n_done + 1))
                continue
            fi
            # Any active (pending/running/completing) job already covering this task id?
            # -r is REQUIRED: without it a pending single-task array prints as 65914679_[13]
            # (brackets) and the _<tid>$ match silently fails -> the guardian re-submitted all
            # 60 cells EVERY hourly tick (600 queued duplicates by tick 10, caught 2026-07-19).
            state=$(squeue -r -u "${USER}" -n marl-chemcasc -h -o "%i %T" 2>/dev/null | awk -v t="_${tid}$" '$1 ~ t {print $2; exit}')
            if [[ -n "${state}" ]]; then
                n_flight=$((n_flight + 1))
                continue
            fi
            # Not done, not in flight -> (re)submit this single cell at 72h.
            sbatch --array="${tid}" --time=3-00:00:00 --job-name=marl-chemcasc "${SELF}" >/dev/null 2>&1 \
                && { echo "[${TS}] submitted task ${tid} (${cell})" >> "${LOG}"; n_sub=$((n_sub + 1)); } \
                || echo "[${TS}] WARN: sbatch failed for task ${tid} (${cell})" >> "${LOG}"
        done
    done
    echo "[${TS}] tick: done=${n_done}/60 in_flight=${n_flight} submitted_this_tick=${n_sub}" >> "${LOG}"
    echo "${n_done}"
}

# PRIORITY MODE (2026-08-02). The 9 chemistry-cascade cells of seeds 42-44 were starved: every
# job of mine carries the same priority (1709956), so SLURM breaks ties by job id and the main
# array (65249080) always won over the chemcasc jobs (65914688+) -- they would only have run
# after all 332 remaining main-array tasks. Everything else is therefore HELD so those 9 take
# the next free GPUs. This releases the held jobs as soon as the 9 are done: without it the
# whole run would stall behind the hold forever.
release_if_priority_done() {
    local n=0 s st held grp
    for s in 42 43 44; do
        for st in cascade_1st_no_meta maps meta_cascade_both; do
            [[ -s "${OB}/chemistry/setting-${st}/seed-${s}/metrics.json" ]] && n=$((n + 1))
        done
    done
    (( n < 9 )) && return 0
    held=$(squeue -r -u "${USER}" -h -t PENDING -o "%i %r" 2>/dev/null | awk '$2 ~ /JobHeldUser/ {print $1}')
    [[ -z "${held}" ]] && return 0
    printf '%s\n' ${held} | xargs -r -n 50 echo | while read -r grp; do
        scontrol release "$(echo "${grp}" | tr ' ' ',')" 2>/dev/null
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] priority cells (seeds 42-44) all done -> released $(printf '%s\n' ${held} | wc -l) held jobs" >> "${LOG}"
}

# Queue the successor at the START (survives any failure later in this job).
SUCCESSOR_ID=$(sbatch --parsable --begin=now+86400 "${GUARDIAN_SELF}" 2>/dev/null) \
    || echo "$(date '+%Y-%m-%d %H:%M:%S') WARN: could not queue successor guardian" >> "${LOG}"

for _ in $(seq 1 24); do
    DONE=$(tick)
    release_if_priority_done
    if (( DONE >= 60 )); then
        echo "$(date '+%Y-%m-%d %H:%M:%S') ALL 60 chemistry-cascade cells complete — guardian stopping." >> "${LOG}"
        [[ -n "${SUCCESSOR_ID:-}" ]] && scancel "${SUCCESSOR_ID}" 2>/dev/null
        break
    fi
    sleep 3600
done
