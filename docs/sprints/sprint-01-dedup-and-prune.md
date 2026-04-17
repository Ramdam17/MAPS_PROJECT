# Sprint 01 — Dedup & prune

**Status:** ⚪ planned
**Branch:** `refactor/clean-rerun` (continuation)
**Owner:** Rémy Ramadour
**Est. effort:** 1-2 days
**Depends on:** Sprint 00 ✅

---

## Context

The audit found ~5k LOC of dead/duplicated code across the repo. Removing it **before** extracting the MAPS core reduces surface area, eliminates ambiguity about "which copy is the real one", and makes the Phase 3 refactor tractable.

## Objectives

1. One copy of `energy_tracker.py` (vs. 4 byte-identical today)
2. One copy of `MinAtar/` (vs. 2 today: `SARL/` and `SARL_CL/`)
3. Remove accidental/shadow files
4. Consolidate directory structure — move vendored deps under `external/`

## Tasks

### 1.1 — Kill duplicates
- [ ] Diff-verify `energy_tracker.py` is byte-identical in the 4 locations
- [ ] Move to `src/maps/utils/energy_tracker.py`
- [ ] Delete copies in `AGL/`, `BLINDSIGHT/`, `MARL/MAPPO-ATTENTIOAN/`, `SARL/MinAtar/examples/`
- [ ] Update imports in Blindsight_TMLR.py, AGL_TMLR.py, SARL examples, MARL

### 1.2 — Shadow files
- [ ] Delete `MARL/MAPPO-ATTENTIOAN/onpolicy/algorithms/happo/happo_trainer(1).py` (accidental download)
- [ ] Delete `MARL/MAPPO-ATTENTIOAN/onpolicy/envs/some_env_old.py` and `...environment_old.py` (check first)
- [ ] Delete `MARL/MAPPO-ATTENTIOAN/onpolicy/algorithms/utils/cnn_original.py`
- [ ] Resolve `SARL/MinAtar/examples/maps_v1.py` vs `maps_v2.py`:
    - [ ] diff v1 vs v2; confirm v2 is newer/correct
    - [ ] delete v1, rename v2 → `maps.py`

### 1.3 — MinAtar consolidation
- [ ] Diff-verify `SARL/MinAtar/` vs `SARL_CL/MinAtar/` — ensure both are the same `kenjyoung/MinAtar` v1.0.15
- [ ] Move to `external/MinAtar/`
- [ ] Delete `SARL/MinAtar/build/` and `SARL_CL/MinAtar/build/` (artifacts)
- [ ] Delete `SARL/MinAtar/MinAtar.egg-info/`
- [ ] Delete `SARL_CL/MinAtar/` (after consolidation)
- [ ] Update `pyproject.toml` `[tool.uv.sources]`: `minatar = { path = "external/MinAtar", editable = true }`
- [ ] Smoke test: `uv run python -c "from minatar import Environment; Environment('breakout')"`

### 1.4 — Plot_scaling_discrimination shadow (Blindsight)
- [ ] Remove duplicate `plot_scaling_discrimination` function at `BLINDSIGHT/Blindsight_TMLR.py:1521` (first definition, silently shadowed by line 1677)
- [ ] (Lightweight in-place edit — this file gets split in Sprint 04 anyway)

### 1.5 — METTA relocation
- [ ] `git mv METTA/metta_MAPS external/METTA`
- [ ] Ensure its own `pyproject.toml`, `uv.lock`, `AGENTS.md`, `CLAUDE.md` move with it
- [ ] Add a top-level `external/METTA/README_INTEGRATION.md` documenting: "tests run via their own `uv sync` in this subdir, do not mix with root `maps` env"

### 1.6 — Final sweep
- [ ] Remove any remaining `__pycache__` committed in tree (should already be gitignored)
- [ ] Verify `du -sh .git` stays under 60 MB
- [ ] Verify `uv sync --extra blindsight --extra agl --extra sarl` still succeeds

## Files touched (approximate)

| Action | Count | Paths |
|--------|------:|-------|
| Delete | ~8 | shadow files, old duplicates |
| Move | 1 copy × 4 | `energy_tracker.py` |
| Move | 1 tree | `METTA/metta_MAPS` → `external/METTA` |
| Move | 1 tree | `SARL/MinAtar` → `external/MinAtar` |
| Delete tree | 2 | `SARL/MinAtar/build`, `SARL_CL/MinAtar` |

## Definition of Done

- `find . -name energy_tracker.py -not -path "./external/*" -not -path "./.venv/*"` returns exactly **1 path** (`src/maps/utils/energy_tracker.py`)
- `find . -name MinAtar -type d -not -path "./external/*"` returns **0 paths**
- No `happo_trainer(1).py`, `cnn_original.py`, `*_old.py` anywhere
- `uv run pre-commit run --all-files` still passes
- `uv sync --extra all-local` still succeeds
- Commit message per `git-discipline` conventions, single commit per sub-task

## Risks

- Blindsight_TMLR.py still imports `from energy_tracker import ...` — breaks if moved without updating imports. Mitigation: update both in the same atomic commit.
- `maps_v1.py` and `maps_v2.py` may not be strictly deletable — CI/reproduction scripts may reference one or the other. Check with `grep -r "maps_v1\|maps_v2" .sh` before deleting.
