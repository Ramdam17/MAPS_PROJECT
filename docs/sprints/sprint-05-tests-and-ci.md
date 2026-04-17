# Sprint 05 — Tests + CI

**Status:** ⚪ planned
**Branch:** `feat/tests-ci`
**Owner:** Rémy Ramadour
**Est. effort:** 1-2 days
**Depends on:** Sprint 04

---

## Context

After Sprint 04, we have a modular, config-first codebase but no automated safety net. This sprint adds the test suite that'll guard against behavior regressions during reproduction sprints (06-08).

We target **3 pyramid levels**:
1. **Unit** (fast, <1 min total) — components in isolation
2. **Numerical** (fast, <5 min) — invariants and parity
3. **Reproduction smoke** (slow, ~10 min) — short training runs that hit expected loss trajectories within tolerance

Full reproduction tests (paper z-scores with multi-seed) live in Sprint 06-08, not here.

## Objectives

1. Unit coverage ≥ 80% on `src/maps/components` and `src/maps/utils`
2. GitHub Actions: ruff + pytest (fast tiers) on every push
3. Pre-push hook: same, locally, under 60 s

## Tasks

### 5.1 — Unit tests
- [ ] `tests/unit/components/test_cascade.py`:
    - Shape preservation: `out.shape == in.shape`
    - α=1 ⇒ output matches single forward pass (bit-identical, no iteration effect)
    - n_iter=0 degenerate case raises
    - Gradient flows to input
    - State reset between samples is observable
- [ ] `tests/unit/components/test_second_order.py`:
    - Comparator shape: `C = X - Y_hat`
    - Wagering output has 2 units
    - Dropout is active in `.train()`, off in `.eval()`
    - Gradient flows through wagering head
- [ ] `tests/unit/components/test_losses.py`:
    - `contrastive_loss`: zero when `z_i == z_j` and `N=1`; matches reference SimCLR value on a fixed toy input
    - `wagering_bce_loss`: numerically equals `F.binary_cross_entropy_with_logits` for uniform weights
    - `distillation_loss`: respects `reduction=...`
- [ ] `tests/unit/networks/test_first_order_mlp.py`:
    - Shape in/out, param count matches (hidden=40, layers=2) ⇒ expected param count

### 5.2 — Numerical / parity tests
- [ ] `tests/parity/test_blindsight_reference.py` (scaffolded in Sprint 02)
- [ ] `tests/parity/test_agl_reference.py`
- [ ] `tests/numerical/test_seeding.py`:
    - `set_all_seeds(42)` + forward + `set_all_seeds(42)` + forward ⇒ bit-identical
    - CPU vs. MPS gives same shape/dtype (numerical tolerance 1e-5)

### 5.3 — Reproduction smoke tests (`@pytest.mark.slow`)
- [ ] `tests/reproduction/test_blindsight_smoke.py`:
    - 2-epoch training run with seed=42
    - Assert final loss < threshold (calibrated from a local reference run)
    - Runs in <3 min on CPU
- [ ] `tests/reproduction/test_agl_smoke.py` — same pattern
- [ ] `tests/reproduction/test_sarl_smoke.py`:
    - 1000 training frames on Breakout
    - Assert env loads, epsilon decays, replay buffer fills
    - ~2 min

### 5.4 — GitHub Actions
- [ ] `.github/workflows/ci.yaml`:
    - Matrix: ubuntu-latest only (MARL tests gated linux_only, MPS gated via `pytest.mark.gpu`)
    - Python 3.12
    - `uv sync --extra blindsight --extra agl --extra sarl --extra dev`
    - `uv run ruff check .`
    - `uv run pytest -m "not slow and not gpu"`
- [ ] `.github/workflows/slow-tests.yaml`:
    - Manual trigger (`workflow_dispatch`)
    - Runs `pytest -m slow` (reproduction smoke)

### 5.5 — Coverage
- [ ] `pytest-cov` configured in pyproject
- [ ] `coverage.xml` as CI artifact
- [ ] Target ≥ 80% on `src/maps/components` and `src/maps/utils`
- [ ] No coverage requirement yet on `src/maps/experiments/*` (reproduction tests cover those)

### 5.6 — Pre-push hook
- [ ] Extend `.pre-commit-config.yaml` with a local `pre-push` stage:
    - `uv run pytest -m "not slow and not gpu" --timeout=60`
- [ ] Install: `uv run pre-commit install --hook-type pre-push`

## Definition of Done

- `uv run pytest -m "not slow and not gpu"` runs in < 60 s and passes
- `uv run pytest -m slow` runs in < 15 min and passes
- GitHub Actions green on `refactor/clean-rerun` → merge PR to main
- `uv run pytest --cov=src/maps/components --cov-fail-under=80` passes
- Badge in README: ![CI](https://github.com/Ramdam17/MAPS_PROJECT/actions/workflows/ci.yaml/badge.svg)

## Risks

- MPS nondeterminism: `torch.use_deterministic_algorithms(True, warn_only=True)` may not be enough on Apple Silicon. Mitigation: parity tests run on CPU only, MPS gated with `@pytest.mark.gpu` + tolerance.
- Reference values for smoke tests must come from a trusted baseline run. Do 3 local runs before fixing a threshold.
