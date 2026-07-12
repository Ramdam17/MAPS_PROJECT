#!/usr/bin/env bash
# MARL run watcher — hourly, self-rescheduling, disconnection-proof (runs as a SLURM CPU job,
# not tied to a login session). Each hour it:
#   1. counts completed cells (metrics.json present) out of 480,
#   2. appends a timestamped progress breakdown to $SCRATCH/marl_reports/progress.txt,
#   3. bundles all completed metrics.json into $SCRATCH/marl_reports/marl_metrics_latest.tar.gz
#      (a fresh full snapshot to download / send progressively to Guillaume & Natalie),
#   4. reschedules itself in 1h — until all 480 are done, then stops.
#
# Start:  sbatch scripts/slurm/marl_watch.sh
# Stop:   scancel --name=marl-watch          (cancels the pending next tick)
# Read:   cat $SCRATCH/marl_reports/progress.txt   |   scp .../marl_reports/marl_metrics_latest.tar.gz

#SBATCH --job-name=marl-watch
#SBATCH --account=def-gdumas85_cpu
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/marl-watch-%j.out
#SBATCH --error=logs/slurm/marl-watch-%j.out

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

OUT_BASE="${SCRATCH:-${REPO_ROOT}/outputs}/maps/outputs/marl"
REP="${SCRATCH:-${REPO_ROOT}/outputs}/marl_reports"
mkdir -p "${REP}"
TOTAL=480
INTERVAL=3600

TS=$(date '+%Y-%m-%d %H:%M:%S')
DONE=$(find "${OUT_BASE}" -name metrics.json 2>/dev/null | wc -l)

COMPLETE_SEEDS=""
{
    echo "==================== [${TS}] watcher job=${SLURM_JOB_ID:-?} ===================="
    echo "MARL production: ${DONE} / ${TOTAL} cells complete"
    echo "--- per-seed completion (24 cells = 4 substrates x 6 settings each) ---"
    for s in $(seq 42 61); do
        c=$(find "${OUT_BASE}" -path "*/seed-${s}/metrics.json" 2>/dev/null | wc -l)
        if (( c == 24 )); then COMPLETE_SEEDS="${COMPLETE_SEEDS} ${s}"; fi
        printf "  seed %2d: %2d/24%s\n" "${s}" "${c}" "$( ((c == 24)) && echo '   <-- COMPLETE (ready to send)' )"
    done
    echo "  => complete seeds:${COMPLETE_SEEDS:- none yet}"
    echo "--- completed per substrate/setting ---"
    find "${OUT_BASE}" -name metrics.json 2>/dev/null \
        | sed "s|${OUT_BASE}/||; s|/seed-[0-9]*/metrics.json||" | sort | uniq -c
} >> "${REP}/progress.txt"

# Per-seed bundle (built once, when a seed's 24 cells are all done) — send these one by one.
for s in ${COMPLETE_SEEDS}; do
    tarf="${REP}/marl_seed-${s}.tar.gz"
    if [[ ! -f "${tarf}" ]]; then
        ( cd "$(dirname "${OUT_BASE}")" \
          && tar czf "${tarf}.tmp" marl/*/*/seed-${s}/metrics.json 2>/dev/null \
          && mv "${tarf}.tmp" "${tarf}" ) || true
    fi
done

# Fresh full bundle of everything completed so far (small: ~0.75 MB per cell).
if (( DONE > 0 )); then
    ( cd "$(dirname "${OUT_BASE}")" && \
      tar czf "${REP}/marl_metrics_latest.tar.gz.tmp" marl/*/*/*/metrics.json 2>/dev/null \
      && mv "${REP}/marl_metrics_latest.tar.gz.tmp" "${REP}/marl_metrics_latest.tar.gz" ) || true
fi

# Reschedule until the run is complete.
if (( DONE < TOTAL )); then
    sbatch --begin=now+${INTERVAL} "${REPO_ROOT}/scripts/slurm/marl_watch.sh" >/dev/null 2>&1 \
        || echo "[${TS}] WARN: failed to reschedule watcher (resubmit manually)"
else
    echo "[${TS}] ALL ${TOTAL} cells complete — watcher stopping." >> "${REP}/progress.txt"
fi
