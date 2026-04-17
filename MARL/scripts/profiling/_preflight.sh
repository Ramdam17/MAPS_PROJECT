#!/bin/bash
# Shared preflight for MARL profiling scripts.
# - Resolves a working python (prefers $PROJECT_HOME/.venv/bin/python, then uv)
# - Verifies the MARL deps are importable (torch + meltingpot + dmlab2d)
# - Exports $PY for the caller
#
# Usage (sourced, not executed):
#   source "$(dirname "$0")/_preflight.sh"
#   "$PY" some_script.py ...

# shellcheck shell=bash

# Locate repo root (two levels up from this file: MARL/scripts/profiling -> repo).
_PREFLIGHT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_PREFLIGHT_DIR}/../../.." && pwd)"

echo "[preflight] repo root: ${REPO_ROOT}"

# ───── Resolve python ──────────────────────────────────────────────────────────
PY=""
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PY="${REPO_ROOT}/.venv/bin/python"
    echo "[preflight] using venv python: ${PY}"
elif command -v uv >/dev/null 2>&1; then
    # uv run python — spawns a subshell each call but works without manual activation.
    PY="uv run --project ${REPO_ROOT} python"
    echo "[preflight] using 'uv run python' (no .venv found at ${REPO_ROOT}/.venv)"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
    echo "[preflight] WARNING: no .venv and no uv — falling back to system python3"
else
    echo "[preflight] ERROR: no python available."
    echo "             Run: cd ${REPO_ROOT} && uv sync --extra marl --extra dev"
    exit 1
fi

echo "[preflight] python version: $(${PY} --version 2>&1)"

# ───── Check MARL deps ─────────────────────────────────────────────────────────
# torch is cross-platform; meltingpot + dmlab2d are Linux-only (see CLAUDE.md).
_check_import() {
    local mod="$1"
    if ! ${PY} -c "import ${mod}" 2>/dev/null; then
        echo "[preflight] missing module: ${mod}"
        return 1
    fi
    return 0
}

MISSING=()
_check_import torch        || MISSING+=("torch")
_check_import meltingpot   || MISSING+=("meltingpot")
_check_import dmlab2d      || MISSING+=("dmlab2d")

if (( ${#MISSING[@]} > 0 )); then
    echo ""
    echo "[preflight] ERROR: missing required modules: ${MISSING[*]}"
    echo ""
    case "$(uname -s)" in
        Darwin)
            echo "  You are on macOS. meltingpot/dmlab2d are Linux-only (see CLAUDE.md)."
            echo "  For MARL profiling, submit the profile_setting*.sh to Narval via:"
            echo "    sbatch MARL/sbatch/train_maps_marl.sbatch  (adapt for profiling)"
            echo "  Or run from a Linux host with: uv sync --extra marl --extra dev"
            ;;
        Linux)
            echo "  On Linux, install MARL extras with:"
            echo "    cd ${REPO_ROOT} && uv sync --extra marl --extra dev"
            ;;
        *)
            echo "  Install MARL extras: uv sync --extra marl --extra dev"
            ;;
    esac
    exit 1
fi

echo "[preflight] all MARL deps importable ✓"
export PY
