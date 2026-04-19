#!/usr/bin/env bash
# Shared setup sourced by every SLURM script (smoke, bench, array, aggregate).
#
# Guarantees a consistent compute-node environment:
#   - StdEnv/2023 + python/3.12 + cuda/12.6 modules
#   - uv on PATH (installed once under $HOME/.local/bin)
#   - pre-synced .venv activated (uv sync must have run on the login node;
#     compute nodes on DRAC/Tamia have no outbound internet)
#   - PYTHONUNBUFFERED=1 so `logging` flushes to stdout in real time
#   - logs/slurm/ exists at the project root
#   - SIGTERM/SIGUSR1 traps log the interruption reason before exit
#
# Usage (at the top of every sbatch script, after the #SBATCH block):
#
#     source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# The script then runs in the configured env; nothing else to do.

set -euo pipefail

# ── Repository root ────────────────────────────────────────────────────────
# sbatch copies the script to /var/spool/slurmd/job<N>/slurm_script, so
# BASH_SOURCE is useless for locating the repo. SLURM_SUBMIT_DIR is the dir
# from which `sbatch` was invoked — we always submit from the project root,
# so that's the repo. Fallback to cwd for interactive `source common.sh`.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export REPO_ROOT

# Sanity: this must be a MAPS checkout (has config/paths.yaml at root).
if [[ ! -f "${REPO_ROOT}/config/paths.yaml" ]]; then
    echo "[common.sh] REPO_ROOT=${REPO_ROOT} does not contain config/paths.yaml — submit sbatch from project root, or set SLURM_SUBMIT_DIR explicitly." >&2
    exit 78  # EX_CONFIG
fi

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

# ── Modules (Tamia / DRAC CCDB) ────────────────────────────────────────────
# StdEnv/2023 is sticky and usually already loaded; redo for idempotence.
module load StdEnv/2023 python/3.12 cuda/12.6

# ── uv on PATH ─────────────────────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || {
    echo "[common.sh] uv not found at \$HOME/.local/bin/uv — install via 'curl -LsSf https://astral.sh/uv/install.sh | sh' on login node first." >&2
    exit 78  # EX_CONFIG
}

# ── Python env ─────────────────────────────────────────────────────────────
# Compute nodes have no internet; the venv must be pre-synced on login.
if [[ ! -f "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "[common.sh] .venv missing at ${REPO_ROOT}/.venv — run 'uv sync --extra sarl --extra dev' on the login node first." >&2
    exit 78
fi

# We always invoke via `uv run --offline` so we explicitly fail if something
# tries to resolve/install. Real-time logging for long-running SARL cells.
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1

# ── Signal traps ───────────────────────────────────────────────────────────
# SLURM sends SIGTERM when --time is exceeded (or pre-emption); SIGUSR1 is
# our --signal option for gentle warn-before-kill. Log which, with wall-clock.
_common_sh_start_s=${SECONDS}
_common_log_exit() {
    local signal=$1
    local wall=$(( SECONDS - _common_sh_start_s ))
    echo "[common.sh] received ${signal} after ${wall}s wall-clock — job_id=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?}" >&2
}
trap '_common_log_exit SIGTERM' TERM
trap '_common_log_exit SIGUSR1' USR1

# ── Summary line (visible in .out) ─────────────────────────────────────────
echo "[common.sh] ready — REPO_ROOT=${REPO_ROOT} python=$(uv run --offline python --version 2>&1) torch=$(uv run --offline python -c 'import torch; print(torch.__version__)' 2>&1) cuda_avail=$(uv run --offline python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
