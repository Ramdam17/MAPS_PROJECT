# Sprint 00 — Remise en route MAPS

**Status:** 🟡 in progress
**Branch:** `refactor/clean-rerun`
**Owner:** Rémy Ramadour
**Started:** 2026-04-17

---

## Context

MAPS_PROJECT is a fork of Juan David Vargas's lab codebase. The original author (PhD student) has left and the published results are no longer reproducible from the current state of the code. Goal: clean + reproduce, without scope creep toward new research.

**Non-goals:**
- Submission to Nature Machine Intelligence (deferred — separate effort with Juan)
- Reimplementation of MAPS theory
- METTA-AI feature development

---

## Problem Inventory (from @tech-debt + @config-auditor audits)

### 🔴 Blocker — security
- [x] AWS keys leaked in README.md (revoked via Juan — TBD)
- [x] wandb tokens in MARL shell scripts
- [ ] Git history rewrite (filter-repo) + force-push to fork

### 🔴 Blocker — reproducibility
- [ ] Zero seed control anywhere in Blindsight/AGL/SARL scripts
- [ ] `requirements.txt` unusable (416 lines, duplicated, stdlib packages shadowed)
- [ ] Hardcoded absolute paths in 5 shell scripts (3 different authors' machines)

### 🟠 Structural
- [ ] `energy_tracker.py` duplicated 4× byte-identical
- [ ] MAPS components re-implemented across Blindsight/AGL/SARL
- [ ] Two monoliths: Blindsight_TMLR.py (2419 LOC), AGL_TMLR.py (2785 LOC)
- [ ] 315 `print()` calls, no `logging` usage
- [ ] ~20 bare `except:` / `except Exception: pass`
- [ ] No tests, no CI, no .gitignore (committed `__pycache__`, `awscliv2.zip`, `wandb/`, build artifacts)
- [ ] `plot_scaling_discrimination` defined 2× in Blindsight (silent shadow)
- [ ] `happo_trainer(1).py`, `environment_old.py`, `cnn_original.py` — accidental/shadow files

---

## Phase Plan

### Phase 0 — Security (day 1) ✅ *in progress*
- [x] Create branch `refactor/clean-rerun`
- [x] Scrub secrets from tracked files
- [x] Add `.gitignore`
- [ ] `git filter-repo` history rewrite + force-push
- [ ] Notify Juan + Zahra to rotate keys
- [ ] Add `detect-secrets` pre-commit hook

### Phase 1 — Cadrage (day 1-2)
- [x] `docs/TODO.md` populated
- [x] `CLAUDE.md` written
- [x] `docs/sprints/sprint-00-remise-en-route.md` (this file)
- [ ] `docs/reproduction/experiment_matrix.md` — paper z-score targets
- [ ] Decision log for any deviation from paper

### Phase 2 — Environnement reproductible (day 3)
- [ ] Single `pyproject.toml` at root with `[project.optional-dependencies]`
- [ ] `uv.lock` committed
- [ ] Pin versions from actual `import`s (not from old `pip freeze`)
- [ ] Remove stdlib shadows (`typing`, `dataclasses`, `contextvars`, `configparser`)
- [ ] `uv sync --extra blindsight --extra agl --extra sarl` works on Mac
- [ ] Document MARL/METTA = Linux-only in README

### Phase 3 — Refacto config-first + modular (day 4-7)
- [ ] Create `src/maps/` package skeleton
- [ ] Move `energy_tracker.py` to `src/maps/utils/` (1 copy)
- [ ] Extract `src/maps/components/{second_order,cascade,losses}.py`
- [ ] Create `src/maps/utils/seeding.py` + wire it into every entry point
- [ ] Create `src/maps/utils/logging_setup.py` + replace all `print()`
- [ ] Create `config/maps.yaml` with canonical constants
- [ ] Create `config/paths.yaml` (mac / narval / rorqual)
- [ ] Create `config/training/{blindsight,agl,sarl_dqn,sarl_cl,marl_mappo}.yaml`
- [ ] Split monoliths — Blindsight → `src/maps/experiments/blindsight/*.py`, same for AGL
- [ ] Unified CLI: `scripts/train_{blindsight,agl,sarl,sarl_cl,marl}.py` with `typer`
- [ ] Deduplicate: remove `maps_v2.py` (keep v1), `happo_trainer(1).py`, `environment_old.py`, `cnn_original.py`, double `build/lib/minatar/`
- [ ] Move `METTA/metta_MAPS/` → `external/METTA/`, MinAtar → `external/MinAtar/` (1 copy)

### Phase 4 — Tests + CI (day 8)
- [ ] `tests/unit/test_cascade.py` — invariants (α=0 ⇒ single-pass; α=1 ⇒ immediate convergence)
- [ ] `tests/unit/test_second_order.py` — shape, gradient flow, wagering output
- [ ] `tests/numerical/test_seeding.py` — identical seed ⇒ identical output
- [ ] `tests/reproduction/test_blindsight_smoke.py` — short training run matches expected loss trajectory
- [ ] GitHub Actions: `uv run ruff check + pytest` on push
- [ ] `pre-commit` with ruff + detect-secrets + no-committed-large-files

### Phase 5 — Reproduction sur Narval (1-2 weeks compute)
- [ ] Port to Compute Canada — SLURM job scripts in `scripts/slurm/`
- [ ] Run Blindsight + AGL with seeds 1-5 × 6 settings (local Mac OK)
- [ ] Run SARL × 5 envs × 6 settings × 3 seeds (Narval)
- [ ] Run MARL × 4 substrates × 6 settings × 3 seeds (Narval)
- [ ] Compare z-scores vs `docs/reproduction/experiment_matrix.md`
- [ ] Document deviations in `docs/reproduction/report.md`

### Phase 6 — Documentation (day 20+)
- [ ] README rewrite (install, quickstart Mac vs Narval, reproduction guide)
- [ ] NumPy-style docstrings on all public API of `src/maps/`
- [ ] `notebooks/00_maps_components_demo.ipynb`
- [ ] `notebooks/01-04_{domain}_walkthrough.ipynb`
- [ ] Final `docs/reports/sprint-00-report.md` + PDF

---

## Open Questions

1. Do we keep METTA-AI in scope for reproduction, or only "it installs"? (leaning: only installs)
2. Which MARL substrates to prioritise? Paper reports on 4; compute budget may allow only 2.
3. Seed count for reproduction: paper says 10, but Narval quota may force 3-5.

---

## Success Criteria (Definition of Done)

- `uv sync --extra all-local && uv run pytest` passes on clean Mac
- Blindsight z-score within ±2σ of paper value (0.97 ± 0.02)
- AGL z-scores within ±2σ of paper
- At least 2 SARL environments reproduce within ±2σ on Narval
- Zero hardcoded paths, zero committed secrets, zero `print()` in `src/maps/`
- Pre-commit hook blocks secret/path regressions
