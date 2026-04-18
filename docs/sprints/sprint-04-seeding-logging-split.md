# Sprint 04 — Seed control + logging + split monoliths

**Status:** 🟡 in progress
**Branch:** `refactor/blindsight-train` (merged as `325c73b`) → next: `refactor/monoliths`
**Owner:** Rémy Ramadour
**Est. effort:** 2-3 days
**Depends on:** Sprint 03 ✅

---

## Progress (2026-04-17)

- ✅ **4.1 Seeding** — landed earlier as part of Sprint 03 (`src/maps/utils/seeding.py` + unit tests)
- ✅ **4.3 Blindsight split** — `src/maps/experiments/blindsight/{data.py,trainer.py}` shipped, parity-tested vs reference (atol=1e-5, 4/4 factorial settings) — merged as `325c73b`
- ❌ **4.2 Logging setup** — `src/maps/utils/logging_setup.py` not created; T201 lint rule not enforced
- ❌ **4.4 AGL split** — 2785 LOC monolith still at `AGL/AGL_TMLR.py`
- ❌ **4.5 SARL split** — monoliths still at `SARL/MinAtar/examples/`
- ❌ **4.6 SARL+CL split** — monoliths still at `SARL_CL/`
- ❌ **4.7 Legacy delete** — `BLINDSIGHT/`, `AGL/`, `SARL/`, `SARL_CL/` roots untouched
- ❌ **4.8 `except:` cleanup** — not audited

Sprint is ~25% complete. Remaining scope = AGL split (the next parity-critical piece) + logging audit + legacy delete of Blindsight. SARL / SARL+CL splits may be deferred to a sub-sprint given their size.

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

### 4.2 — Logging setup
- [ ] `src/maps/utils/logging_setup.py`:
    - `configure_logging(level, log_file: Path | None)` — stdlib logging
    - Formatter includes `%(asctime)s`, `%(name)s`, `%(levelname)s`
    - No silencing of progress bars (tqdm, etc.) — user's CLAUDE.md rule
- [ ] Replace `print(` with `log.info(` or `log.debug(` across new `src/maps/` code (legacy code stays for now — it's deleted in 4.4)
- [ ] Lint rule: add `"T201"` (no print) to `[tool.ruff.lint]` for `src/maps/`

### 4.3 — Split Blindsight monolith
- [ ] `src/maps/experiments/blindsight/data.py` — stimulus generation
- [ ] `src/maps/experiments/blindsight/model.py` — wraps `maps.components` + `maps.networks`
- [ ] `src/maps/experiments/blindsight/train.py` — training loop, takes DictConfig
- [ ] `src/maps/experiments/blindsight/evaluate.py` — metrics (detection acc, z-scores)
- [ ] `src/maps/experiments/blindsight/plots.py` — figures only, no side effects on disk unless explicit `output_dir`
- [ ] `scripts/train_blindsight.py` — typer entry point (scaffold from Sprint 03)

### 4.4 — Split AGL monolith
- [ ] Same 5-file split as above under `src/maps/experiments/agl/`
- [ ] `scripts/train_agl.py` typer entry

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

### 4.8 — `except` cleanup
- [ ] Replace bare `except:` and `except Exception: pass` with specific exception types + log
- [ ] Any genuinely-ignored exception gets a comment explaining why

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
