# Sprint 04 — Seed control + logging + split monoliths

**Status:** 🟡 in progress — perceptual domain + infra done; RL domains deferred to sub-sprints
**Branch:** `refactor/blindsight-train` (`325c73b`) → `refactor/agl-train` (`2604e7b`) → next: `refactor/sarl-train`
**Owner:** Rémy Ramadour
**Est. effort:** 2-3 days (original) — ~60% shipped
**Depends on:** Sprint 03 ✅

---

## Progress (2026-04-17)

- ✅ **4.1 Seeding** — landed earlier as part of Sprint 03 (`src/maps/utils/seeding.py` + unit tests)
- ✅ **4.2 Logging setup** — `configure_logging(level, log_file)` helper in `src/maps/utils/logging_setup.py`; ruff `T20` rule enforced (scoped ignore for legacy `energy_tracker.py` — see follow-up below) — merged as `2604e7b`
- ✅ **4.3 Blindsight split** — `src/maps/experiments/blindsight/{data.py,trainer.py}` shipped, parity-tested vs reference (atol=1e-5, 4/4 factorial settings) — merged as `325c73b`
- ✅ **4.4 AGL split** — `src/maps/experiments/agl/{data.py,trainer.py}` shipped, parity-tested (atol=1e-5, 4/4 factorial settings); AGL-specific reset-to-initial behavior (ref L751) preserved — merged as `2604e7b`
- ✅ **4.8 `except:` cleanup** — audited `src/maps/`, `scripts/`, `tests/` — zero bare `except:` or `except Exception:` found; ruff `BLE` rule added to prevent regressions
- ❌ **4.5 SARL split** — monoliths still at `SARL/MinAtar/examples/` — **deferred to Sprint 04b**
- ❌ **4.6 SARL+CL split** — monoliths still at `SARL_CL/` — **deferred to Sprint 04b**
- ❌ **4.7 Legacy delete** — `BLINDSIGHT/`, `AGL/`, `SARL/`, `SARL_CL/` roots untouched — **deferred until RL domains split**

### Follow-ups flagged during execution

- `src/maps/utils/energy_tracker.py` still has 13 residual `print()` calls (vendored ~600 LOC energy tracker). T20 lint has a scoped ignore; migration to logging is a standalone follow-up task.

Sprint is ~60% complete on LOC; the two perceptual domains + all infra (seeding, logging, except) are shipped. RL domain splits (4.5/4.6) and legacy delete (4.7) move to a follow-up sprint because of their size and because Sprint 05 (tests+CI) can proceed on what's already extracted.

---

## Context

The two monolithic training scripts (`Blindsight_TMLR.py` 2419 LOC, `AGL_TMLR.py` 2785 LOC) combine: data gen, model def, training loop, evaluation, plotting, and a hardcoded experiment grid. They also contain:
- 315 `print()` calls, 0 `logging` usage
- Zero `torch.manual_seed` / `np.random.seed` / `random.seed`
- ~20 bare `except:` / `except Exception: pass` (Sprint 01 covers some MARL ones)

Splitting them into `src/maps/experiments/{blindsight,agl}/` + `scripts/train_*.py` is the last structural blocker before we can run a reproduction.

## Objectives

1. **Seed control** mandatory at every entry point
2. Replace all `print()` with `logging.getLogger(__name__).{info,debug,warning}`
3. No more monoliths: data / model / training / eval in separate modules

## Tasks

### 4.1 — Seeding utility
- [ ] `src/maps/utils/seeding.py`:
    ```python
    def set_all_seeds(seed: int, deterministic: bool = True) -> None:
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    ```
- [ ] Unit test: same seed → same sampled tensor (cpu + MPS)
- [ ] Wire into every `scripts/train_*.py` as first line after config load

### 4.2 — Logging setup ✅
- [x] `src/maps/utils/logging_setup.py`: `configure_logging(level, log_file: Path | None)` with optional file handler
- [x] Formatter includes `%(asctime)s`, `%(name)s`, `%(levelname)s`
- [x] Never silences third-party loggers or progress bars (CLAUDE.md rule)
- [x] Lint rule: `"T20"` added to `[tool.ruff.lint]` with scoped ignore for legacy `energy_tracker.py`
- [x] Wired into `scripts/run_agl.py` and `scripts/run_blindsight.py`

### 4.3 — Split Blindsight monolith
- [ ] `src/maps/experiments/blindsight/data.py` — stimulus generation
- [ ] `src/maps/experiments/blindsight/model.py` — wraps `maps.components` + `maps.networks`
- [ ] `src/maps/experiments/blindsight/train.py` — training loop, takes DictConfig
- [ ] `src/maps/experiments/blindsight/evaluate.py` — metrics (detection acc, z-scores)
- [ ] `src/maps/experiments/blindsight/plots.py` — figures only, no side effects on disk unless explicit `output_dir`
- [ ] `scripts/train_blindsight.py` — typer entry point (scaffold from Sprint 03)

### 4.4 — Split AGL monolith ✅
- [x] `src/maps/experiments/agl/data.py` — pattern generation (random + grammar A/B), `target_second`, constants
- [x] `src/maps/experiments/agl/trainer.py` — `AGLSetting` + `AGLTrainer` with `build()` / `pre_train()`, preserves reference reset-to-initial behavior
- [x] `scripts/run_agl.py` — typer entry point; saves `losses_{1,2}.npy` + state_dicts + `summary.json`
- [x] Parity tests pass at atol=1e-5 across all 4 factorial settings
- [x] Smoke tests pass (4 settings × 10 epochs)

### 4.5 — Split SARL (maps_v1/v2 → one)
- [ ] `src/maps/experiments/sarl/`: `data.py` (replay buffer), `model.py` (Q-network with cascade), `train.py`, `evaluate.py`
- [ ] `scripts/train_sarl.py`

### 4.6 — SARL+CL
- [ ] `src/maps/experiments/sarl_cl/` — teacher network + distillation loss from `maps.components.losses`
- [ ] `scripts/train_sarl_cl.py`

### 4.7 — Delete legacy
- [ ] Delete `BLINDSIGHT/Blindsight_TMLR.py`
- [ ] Delete `AGL/AGL_TMLR.py`
- [ ] Delete `SARL/MinAtar/examples/maps*.py`, `dqn.py`, `AC_lambda.py` (re-implemented in `src/maps/experiments/sarl/`)
- [ ] Delete now-empty `BLINDSIGHT/`, `AGL/`
- [ ] Leave `MARL/` for Sprint 08 (it's external deps + code we don't own)

### 4.8 — `except` cleanup ✅
- [x] Audited `src/maps/`, `scripts/`, `tests/` — zero bare `except:` or `except Exception:` in managed code
- [x] Ruff `BLE` rule added to prevent regressions (all remaining matches are in `external/` and `MARL/MAPPO-ATTENTION/`, which are ruff-excluded)

## Definition of Done

- `grep -rn "print(" src/maps/` returns 0 hits (or only debug prints in notebooks)
- `grep -rn "except:" src/maps/ scripts/ tests/` returns 0 hits
- Every script in `scripts/` calls `set_all_seeds(cfg.seed)` before any randomness
- `tests/unit/test_seeding.py` passes: two identical runs ⇒ identical loss trajectory
- `uv run pre-commit run --all-files` passes (ruff `T201` enforces print ban)
- `wc -l src/maps/experiments/blindsight/*.py` — each file ≤ 500 LOC
- Smoke test: `uv run python scripts/train_blindsight.py --config config/training/blindsight.yaml --epochs 2 --seed 42` completes on Mac

## Risks

- Behavior drift during split: guard with `tests/parity/` from Sprint 02 — if parity breaks during split, the parity test catches it.
- `tqdm` progress silencing temptation: the rule is "never silence" — use `log.debug` for extra verbosity, not `disable=True`.
