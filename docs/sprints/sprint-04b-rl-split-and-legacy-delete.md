# Sprint 04b — RL domain splits + legacy delete (deferred from Sprint 04)

**Status:** 🟡 in progress — 4.9 shipped, 4.5/4.6/4.7 pending
**Branch:** `refactor/energy-tracker-logging` ✅ → `refactor/sarl-train` (next) → `refactor/legacy-delete`
**Owner:** Rémy Ramadour
**Est. effort:** 3-5 days
**Depends on:** Sprint 04 ✅ (perceptual-domain scope)
**Blocks:** Sprint 07 (SARL reproduction), Sprint 05 §5.3 SARL smoke test

---

## Why this exists

Sprint 04 originally scoped *all* four domain splits (Blindsight, AGL, SARL, SARL+CL) in a single sprint. On execution (2026-04-17), the perceptual half (Blindsight + AGL) + infra (seeding, logging, except-audit) landed cleanly with parity tests at atol=1e-5. The RL half was deferred because:

1. **Size.** `SARL/MinAtar/examples/{maps_v1.py, maps_v2.py}` + `SARL_CL/MinAtar/examples/maps.py` together are ~2500 LOC of DQN / replay-buffer / cascade / distillation code that is *not* a byte-for-byte copy of the perceptual loops — it needs its own parity harness and tests.
2. **Dependencies.** SARL depends on the vendored `external/MinAtar/` and on `gym` — a 2nd interface surface to test against, separate from the NumPy/Torch-only perceptual path.
3. **RL sanity checks.** Parity for RL can't be bit-exact the same way — replay buffers + ε-greedy + target-network updates introduce RNG sources that are hard to fully align with the reference. A reduced-precision parity (loss trajectory within %) + a rollout-length smoke test will be needed.
4. **Unblocking Sprint 05.** The tests/CI sprint can proceed on the perceptual code that's already shipped without waiting for RL. SARL smoke tests (§5.3) will land here in 04b instead.

This is a scope deferral, not a cancellation. All items below are tracked.

---

## Deferred tasks (verbatim from Sprint 04 spec)

### 4.5 — Split SARL (maps_v1/v2 → one)
- [ ] `src/maps/experiments/sarl/`:
    - [ ] `data.py` — replay buffer, transition sampler
    - [ ] `model.py` — Q-network with cascade + second-order wager (MinAtar input)
    - [ ] `trainer.py` — DQN training loop, ε-greedy, target-network sync
    - [ ] `evaluate.py` — rollout metrics (episode reward, per-env score)
- [ ] `scripts/run_sarl.py` — typer entry (mirror `run_blindsight.py` / `run_agl.py`)
- [ ] Collapse `maps_v1.py` and `maps_v2.py` into one parametrized class; `maps_v2` only added `count_parameters` and a `curriculum` flag → expose as kwargs.

### 4.6 — SARL+CL (continual learning)
- [ ] `src/maps/experiments/sarl_cl/` — teacher network + distillation loss from `maps.components.losses`
- [ ] `scripts/run_sarl_cl.py`
- [ ] Do **not** re-implement shared bits — import `QNetwork`, `replay_buffer`, `DistillationLoss`, `world_dynamics` from `maps.experiments.sarl` / `maps.components`. Current SARL_CL duplicates these (see TD-009).

### 4.7 — Delete legacy
- [ ] Delete `BLINDSIGHT/Blindsight_TMLR.py`
- [ ] Delete `AGL/AGL_TMLR.py`
- [ ] Delete `SARL/MinAtar/examples/{maps_v1.py, maps_v2.py, dqn.py, AC_lambda.py}` (re-implemented)
- [ ] Delete now-empty `BLINDSIGHT/`, `AGL/`
- [ ] Leave `MARL/` for Sprint 08

**Gate:** do not delete until 4.5 + 4.6 land and their parity harnesses are green.

### 4.9 — `energy_tracker.py` print → log migration ✅
- [x] Replaced 14 residual `print()` calls in `src/maps/utils/energy_tracker.py` with structured logging:
    - 9 `log.info()` — normal lifecycle and result output (GPU count, project start, energy/emissions totals, save paths)
    - 4 `log.warning()` — invalid states (tracker already running / not running / no data collected)
    - 1 `log.error()` — inside the `except` block when GPU metric parsing fails
- [x] Added module-level `log = logging.getLogger(__name__)`.
- [x] Removed the scoped `T201` per-file ignore in `pyproject.toml` (ruff `T20` now uniform across `src/`).
- [x] Verified module still imports.

---

## Parity testing strategy (RL-specific)

Unlike the perceptual domains, SARL cannot use atol=1e-5 bit-exact parity because:
- Replay buffer insertion order depends on episode boundaries
- ε-greedy sampling consumes RNG unevenly
- Target-network sync cadence interacts with step count

Proposed tiers:
1. **Deterministic-rollout parity:** with `epsilon=0` and a fixed pre-seeded buffer, assert Q-network forward passes match reference at atol=1e-5.
2. **Trajectory parity:** with seeded ε-greedy, assert the first 100 actions match the reference (RNG-coupled).
3. **Loss-trajectory parity:** with full training, assert the loss curve stays within 5% of reference over a 1000-step horizon (burn-in + training).

Tier 1 + 2 live in `tests/parity/`; tier 3 in `tests/reproduction/` with `@pytest.mark.slow`.

---

## Definition of Done

- `src/maps/experiments/sarl/*.py` and `src/maps/experiments/sarl_cl/*.py` each ≤ 500 LOC
- `scripts/run_sarl.py` runs `uv run python scripts/run_sarl.py --env breakout --setting both --n-steps 1000` on Mac CPU
- Three-tier parity tests all pass
- Legacy paths (`BLINDSIGHT/`, `AGL/`, `SARL/`, `SARL_CL/` — except MinAtar which is in `external/`) deleted
- `grep -rn "print(" src/maps/ | wc -l` returns **0** (energy_tracker migrated) ✅
- Ruff `T20` per-file ignore for `energy_tracker.py` removed ✅

---

## Risks

- **MinAtar interface drift.** The vendored MinAtar in `external/MinAtar/` may differ from what `SARL/MinAtar/examples/maps_v*.py` originally imported. Check `env.act()` / `env.state()` signature compatibility before porting.
- **Replay buffer determinism.** PyTorch `torch.randperm` + Python `random.sample` both consume state — set_all_seeds alone may not be enough. May need explicit `torch.Generator` per-buffer.
- **Gym API version.** The paper uses `gym>=0.26`; MinAtar's own API is simpler. Pin `gym` in `pyproject.toml [project.optional-dependencies.sarl]` and don't let `gymnasium` leak in from MARL extras.
