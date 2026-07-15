#!/usr/bin/env bash
# MARL run watcher v2 — disconnection-proof progress/collection (SLURM CPU job).
#
# v1 was a per-HOUR self-resubmit chain; it broke permanently when a single tick hit a
# transient node error ('/usr/bin/env: bash: Input/output error', exit 126) before reaching
# its reschedule line. v2 is robust to that:
#   - one job runs an INTERNAL hourly loop for ~1 day (24 ticks), so handoffs happen daily
#     (25x fewer chances to hit a transient) instead of hourly;
#   - it queues its successor at the START (before doing any work), so a mid-job failure
#     cannot break the chain.
# Each tick: append per-seed progress to $SCRATCH/marl_reports/progress.txt, refresh a
# per-seed marl_seed-<S>.tar.gz once a seed is 24/24, and a full marl_metrics_latest.tar.gz.
#
# Start:  sbatch scripts/slurm/marl_watch.sh
# Stop:   scancel --name=marl-watch          (cancels current + the queued successor)
# Read:   cat $SCRATCH/marl_reports/progress.txt

#SBATCH --job-name=marl-watch
#SBATCH --account=def-gdumas85_cpu
#SBATCH --time=1-01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/marl-watch-%j.out
#SBATCH --error=logs/slurm/marl-watch-%j.out

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"
SELF="${REPO_ROOT}/scripts/slurm/marl_watch.sh"
OB="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl"
REP="${SCRATCH:-${REPO_ROOT}/outputs}/marl_reports"
mkdir -p "${REP}"
TOTAL=480

n_done() { find "${OB}" -name metrics.json 2>/dev/null | wc -l; }

collect() {
    local TS DONE COMPLETE_SEEDS s c tarf
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    DONE=$(n_done)
    COMPLETE_SEEDS=""
    {
        echo "==================== [${TS}] watcher job=${SLURM_JOB_ID:-?} ===================="
        echo "MARL production: ${DONE} / ${TOTAL} cells complete"
        echo "--- per-seed completion (24 cells = 4 substrates x 6 settings each) ---"
        for s in $(seq 42 61); do
            c=$(find "${OB}" -path "*/seed-${s}/metrics.json" 2>/dev/null | wc -l)
            if (( c == 24 )); then COMPLETE_SEEDS="${COMPLETE_SEEDS} ${s}"; fi
            printf "  seed %2d: %2d/24%s\n" "${s}" "${c}" "$( ((c == 24)) && echo '   <-- COMPLETE (ready to send)' )"
        done
        echo "  => complete seeds:${COMPLETE_SEEDS:- none yet}"
        echo "--- completed per substrate/setting ---"
        find "${OB}" -name metrics.json 2>/dev/null \
            | sed "s|${OB}/||; s|/seed-[0-9]*/metrics.json||" | sort | uniq -c
    } >> "${REP}/progress.txt"

    for s in ${COMPLETE_SEEDS}; do
        tarf="${REP}/marl_seed-${s}.tar.gz"
        if [[ ! -f "${tarf}" ]]; then
            ( cd "$(dirname "${OB}")" && tar czf "${tarf}.tmp" marl/*/*/seed-${s}/metrics.json \
              && mv "${tarf}.tmp" "${tarf}" ) >/dev/null 2>&1 || true
        fi
    done
    if (( DONE > 0 )); then
        ( cd "$(dirname "${OB}")" \
          && tar czf "${REP}/marl_metrics_latest.tar.gz.tmp" marl/*/*/*/metrics.json \
          && mv "${REP}/marl_metrics_latest.tar.gz.tmp" "${REP}/marl_metrics_latest.tar.gz" ) >/dev/null 2>&1 || true
    fi
}

# Queue the successor at the START (survives any failure later in this job). Only if not finished.
if (( $(n_done) < TOTAL )); then
    sbatch --begin=now+86400 "${SELF}" >/dev/null 2>&1 || echo "WARN: could not queue successor watcher"
fi

# Internal hourly loop for ~1 day.
for _ in $(seq 1 24); do
    collect
    if (( $(n_done) >= TOTAL )); then
        echo "ALL ${TOTAL} cells complete — watcher chain ends."
        break
    fi
    sleep 3600
done
