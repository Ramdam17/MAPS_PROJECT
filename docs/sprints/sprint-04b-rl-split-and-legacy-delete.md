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

1. **Size.** `external/MinAtar/examples/maps.py` (2721 LOC, SARL) and `SARL_CL/examples_cl/maps.py` (2580 LOC, SARL+CL) together are ~5.3k LOC of DQN / replay-buffer / cascade / distillation code — it needs its own parity harness and tests.
2. **Dependencies.** SARL depends on the vendored `external/MinAtar/` and on `gym` — a second interface surface to test against, separate from the NumPy/Torch-only perceptual path.
3. **RL sanity checks.** End-to-end RL parity can't be bit-exact: replay buffers + ε-greedy + target-network updates introduce RNG sources hard to fully align. Strategy is **architectural-unit parity** (see below), not end-to-end loss-curve parity. Multi-seed learning-curve reproduction is deferred to Sprint 07.
4. **Unblocking Sprint 05.** The tests/CI sprint can proceed on the perceptual code that's already shipped without waiting for RL. SARL smoke tests (§5.3) will land here in 04b instead.

This is a scope deferral, not a cancellation. All items below are tracked.

### Source layout (as of 2026-04-17)

The paper's historical `SARL/MinAtar/examples/maps_v1.py` / `maps_v2.py` have already been consolidated (commit `8aa1138`) and moved (commit `ec5bcb7`). The canonical reference scripts today are:

| Role | Path | LOC |
|------|------|-----|
| SARL MAPS DQN (ours, added to vendored MinAtar) | `external/MinAtar/examples/maps.py` | 2721 |
| SARL+CL MAPS with distillation | `SARL_CL/examples_cl/maps.py` | 2580 |
| Young et al. DQN baseline (vendored, **keep**) | `external/MinAtar/examples/dqn.py` | 443 |
| Young et al. AC-λ baseline (vendored, **keep**) | `external/MinAtar/examples/AC_lambda.py` | 498 |
| SARL+CL AC-λ baseline (ours) | `SARL_CL/examples_cl/AC_lambda.py` | 457 |

**Note:** `external/MinAtar/examples/maps.py` has module-level side effects (instantiates `NvidiaEnergyTracker` and calls `start_tracking()` at import). It also self-shadows the `maps` package name when invoked as a script (its own filename is `maps.py` and `external/MinAtar/examples/` lands on `sys.path[0]`). Neither is a blocker for unit-level parity extraction, but both rule out `import`-based reuse.

---

## Deferred tasks (verbatim from Sprint 04 spec)

### 4.5 — Split SARL (`external/MinAtar/examples/maps.py` → `src/maps/experiments/sarl/`)
- [ ] `src/maps/experiments/sarl/`:
    - [ ] `data.py` — `replay_buffer` + `get_state` transition sampler (refs: paper §3, `maps.py:290-327`)
    - [ ] `model.py` — `QNetwork` (+ `AdaptiveQNetwork`) + `SecondOrderNetwork`, rewritten to use `maps.components.cascade` and `maps.components.second_order` from Sprint 02 (refs: `maps.py:135-281`)
    - [ ] `trainer.py` — DQN training loop, ε-greedy, target-network sync, `train()` inner loop split into `compute_loss` / `step` (refs: `maps.py:663-1091`)
    - [ ] `evaluate.py` — rollout metrics, episode reward, per-env score (refs: `maps.py:890-1091` `evaluation()`)
- [ ] `scripts/run_sarl.py` — typer entry (mirror `run_blindsight.py` / `run_agl.py`); exposes `--game`, `--setting {1..6}`, `--seed`, `--steps`, `--cascade`, `--ema`
- [ ] Keep the paper's 6-setting matrix (not 2×2=4): settings 1-6 where 1=baseline, 2=cascade-only, 3=2nd-order-only, 4=2nd-order+cascade-on-FO, 5=2nd-order+cascade-on-SO, 6=2nd-order+cascade-on-both. See `maps.py:2683-2710`.
- [ ] `count_parameters` and `curriculum` flag → expose as kwargs on `SarlTrainer`.

### 4.6 — SARL+CL (continual learning, `SARL_CL/examples_cl/maps.py` → `src/maps/experiments/sarl_cl/`)
- [ ] `src/maps/experiments/sarl_cl/` — teacher network + `DistillationLoss` (refs: `SARL_CL/examples_cl/maps.py:384-434`)
- [ ] `scripts/run_sarl_cl.py`
- [ ] Do **not** re-implement shared bits — import `QNetwork`, `replay_buffer`, `SecondOrderNetwork`, `world_dynamics` from `maps.experiments.sarl` / `maps.components`. Current SARL_CL duplicates these (see TD-009).
- [ ] `DistillationLoss` + `compute_weight_regularization` go to `maps.components.losses` (shared with any future distillation use).

### 4.7 — Delete / relocate legacy
- [ ] Delete `BLINDSIGHT/Blindsight_TMLR.py` (already re-implemented in `src/maps/experiments/blindsight/`)
- [ ] Delete `AGL/AGL_TMLR.py` (already re-implemented in `src/maps/experiments/agl/`)
- [ ] Delete `external/MinAtar/examples/maps.py` (re-implemented in `src/maps/experiments/sarl/`)
- [ ] Delete `SARL_CL/examples_cl/maps.py` (re-implemented in `src/maps/experiments/sarl_cl/`)
- [ ] Delete `SARL_CL/examples_cl/AC_lambda.py` if Sprint 07 baseline-comparison chart is produced via the paper's figures (TBD — do **not** delete until Sprint 07 scope is locked)
- [ ] **Keep** `external/MinAtar/examples/{dqn.py, AC_lambda.py}` — these are Young et al.'s original MinAtar baselines (pre-MAPS), not re-implementations of our code
- [ ] Delete now-empty `BLINDSIGHT/`, `AGL/`
- [ ] Leave `SARL_CL/` shell if `AC_lambda.py` survives; otherwise delete
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

**Scientific question being answered in 04b:** *Does the refactored package compute the same model outputs, buffer samples, and gradient updates as the paper code?*
**Scientific question deferred to Sprint 07:** *Does the refactored package reproduce the paper's reported z-scores across 5 games × 6 settings × 30 seeds?*

The two questions are separable. 04b proves the refactor is faithful **architecturally**; Sprint 07 uses that guarantee + Narval GPU-time to prove **scientifically** that the numbers match.

End-to-end RL parity can't be bit-exact because replay buffer insertion order depends on episode boundaries, ε-greedy sampling consumes RNG unevenly, and target-network sync cadence interacts with step count. Instead, we decompose into **unit-level parity** where bit-exactness is achievable:

### Reference extraction (prerequisite for all tiers)

Copy (verbatim, with citation comments) the following from `external/MinAtar/examples/maps.py` into `tests/parity/sarl/reference_extracts.py`:

- `QNetwork` (lines 135-183)
- `SecondOrderNetwork` (lines 245-281)
- `replay_buffer` (lines 290-321)
- `get_state` (lines 322-328)
- `target_wager` (lines 514-533)

These become **immutable** — treated as the paper's ground truth for 04b. Strip module-level side effects (no energy tracker). Add a module header banner naming the commit SHA from which the extract was taken.

### Tier 1 — Model forward-pass parity (bit-exact, atol=1e-6)

- Seed once, construct reference `QNetwork` + `SecondOrderNetwork`, dump state_dicts.
- Load identical state_dicts into refactored `maps.experiments.sarl.model`.
- Feed a pinned batch of 32 random MinAtar-shape states (4×10×10) at cascade_iter ∈ {1, 10, 25, 50}.
- Assert `torch.allclose(ref_out, ours, atol=1e-6)`.
- **Captures:** cascade math, second-order comparator, wagering head wiring.

### Tier 2 — Replay buffer sampling parity (bit-exact, indices match)

- Seed Python `random`, NumPy, and PyTorch once.
- Insert the same 10k (s, a, s', r, done, terminated) tuples into reference and refactored buffers.
- Call `.sample(128)` 100 times on each. Assert returned transition indices match exactly.
- **Captures:** `random.sample` vs `torch.randperm` vs `np.random.choice` choice drift.

### Tier 3 — DQN update-rule parity (bit-exact, atol=1e-6)

- With Tier 1 model + Tier 2 buffer pinned, compute one gradient step (Huber loss on 128 sampled transitions) on both reference and refactored.
- Compare: loss scalar, gradient norms per parameter group, updated weights after `optimizer.step()`.
- **Captures:** target-network indexing, `max_a Q(s', a)`, discount math, optimizer state init.

All three tiers live in `tests/parity/sarl/` and run in the **fast tier** (not `@slow`). The full 1000-step loss-curve comparison is **deferred to Sprint 07** where it's a single-seed sanity check before scaling to 30 seeds on Narval.

---

## Definition of Done

- `tests/parity/sarl/reference_extracts.py` exists with verbatim copies of `QNetwork` / `SecondOrderNetwork` / `replay_buffer` / `get_state` / `target_wager`, with citation-banner and source-commit-SHA header
- `src/maps/experiments/sarl/*.py` each ≤ 500 LOC; `src/maps/experiments/sarl_cl/*.py` each ≤ 500 LOC
- `scripts/run_sarl.py` runs `uv run python scripts/run_sarl.py --game breakout --setting 6 --steps 1000 --cascade 50 --ema 25 --seed 1` on Mac CPU
- Three-tier unit parity tests all pass (`tests/parity/sarl/test_tier1_forward.py` / `test_tier2_buffer.py` / `test_tier3_update.py`)
- SARL smoke test (single episode, setting 1, 500 steps) in `tests/smoke/` — fast tier, `<10s` wall
- Legacy paths deleted (see 4.7)
- `grep -rn "print(" src/maps/ | wc -l` returns **0** (energy_tracker migrated) ✅
- Ruff `T20` per-file ignore for `energy_tracker.py` removed ✅
- Sprint 07 gets a clean handoff: refactored code + proven architectural parity + a `config/training/sarl.yaml` with all paper hyperparameters locked

---

## Risks

- **MinAtar interface drift.** The vendored MinAtar in `external/MinAtar/` may differ from what `SARL/MinAtar/examples/maps.py` originally imported. Check `env.act()` / `env.state()` signature compatibility before porting. ✅ Mitigated: `external/MinAtar/` is the very path the paper uses today (commit `ec5bcb7` consolidated the vendored copy).
- **Replay buffer determinism.** PyTorch `torch.randperm` + Python `random.sample` both consume state — `set_all_seeds` alone may not be enough. May need explicit `torch.Generator` per-buffer. Tier 2 test will expose this.
- **Gym API version.** The paper uses `gym>=0.26`; MinAtar's own API is simpler. Pin `gym` in `pyproject.toml [project.optional-dependencies.sarl]` and don't let `gymnasium` leak in from MARL extras. ✅ already pinned.
- **Paper script self-shadows the `maps` package.** `external/MinAtar/examples/maps.py` has the same name as our package. Running it as `python external/MinAtar/examples/maps.py` puts the script's dir on `sys.path[0]`, which makes Python resolve `from maps.utils.energy_tracker import ...` to the script itself. This is the source of the `ModuleNotFoundError: No module named 'maps.utils'; 'maps' is not a package` trap. Once 4.7 deletes that file, the problem disappears.
- **Energy-tracker side effects at import.** `external/MinAtar/examples/maps.py:74-85` instantiates `NvidiaEnergyTracker` and calls `start_tracking()` on import. Do **not** import this module from tests. The reference-extract file in `tests/parity/sarl/` avoids this by copying only the pure classes.
