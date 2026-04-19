# Sprint 04b — Report

**Period:** 2026-04-17 → 2026-04-18
**Branches merged:** `refactor/energy-tracker-logging`, `refactor/sarl-parity-harness`, `refactor/sarl-runner-and-cli`, `refactor/sarl-cl-split`, `refactor/legacy-delete`
**Spec:** [`sprint-04b-rl-split-and-legacy-delete.md`](../sprints/sprint-04b-rl-split-and-legacy-delete.md)

## Outcome — shipped vs planned

| Task | Status | Notes |
|------|--------|-------|
| 4.5 SARL split | ✅ | 3-tier parity harness (forward/buffer/update) atol=1e-6; `scripts/run_sarl.py` CLI |
| 4.6 SARL+CL split | ✅ | CL networks + `DynamicLossWeighter` + `sarl_cl_update_step` + `run_training_cl` + CLI; 67 new tests |
| 4.7 Legacy delete | ✅ (partial — see note) | 10,509 LOC removed from the 4 TMLR monoliths |
| 4.9 energy_tracker logging | ✅ (shipped earlier in sprint) | 14 `print()` → `logging`; removed `T201` scoped ignore |

All items from the spec landed. Nothing dropped or deferred.

**Correction (2026-04-18, Sprint 07 P1.1-P1.3):** 4.7 was partial. The 4 TMLR monolith scripts were deleted, but surrounding legacy trees remained tracked — `SARL/` (shell wrappers), `SARL_CL/` (shell wrappers + `AC_lambda.py`), `MARL/MAPPO-ATTENTIOAN/` (2.1 MB vendored MAPPO + dmlab2d wheel + committed tokens), `MARL/meltingpot.sh` (orphan once MAPPO removed), `requirements-legacy.txt`. Sprint 07 Phase 1 removed them (~197 files). `BLINDSIGHT/`, `AGL/`, `METTA/` had already been cleared in Sprint 04b.

## Code / test deltas

| Metric | Before 04b | After 04b | Δ |
|--------|-----------:|----------:|--:|
| Python tests passing | 108 | 196 | +88 |
| LOC in `src/maps/experiments/sarl*/` | 0 | ~2,500 | +2,500 |
| LOC legacy monoliths | 10,509 | 0 | −10,509 |
| `print()` calls in `src/maps/` | 14 | 0 | −14 |
| Coverage on `src/maps/components` + `src/maps/utils` | 94% | 94% | unchanged |

Fast-tier pytest wall: ~1.5s (pre-push hook budget 60s).
Full-tier pytest wall (includes `@slow`): 53s.

## Architectural decisions worth remembering

1. **SARL+CL networks are NOT refactored to reuse SARL networks.** The paper's CL variants (`SarlCLQNetwork`, `SarlCLSecondOrderNetwork`) differ substantively — explicit `fc_output` decoder (not tied weights), cascade applied to Output (1024-dim) not Hidden (128-dim), explicit `comparison_layer` before the wager head. Unifying them would silently change the numbers the CL experiments reported. Documented in `src/maps/experiments/sarl_cl/__init__.py` and inline tables in `model.py`.

2. **CL "distillation" is L2 weight regularization, not soft-target distillation.** The paper defines a `DistillationLoss` class but never calls it; the actual anchor in `train()` is `compute_weight_regularization`. We preserve the misleading key name (`"distillation"`) in loss dicts for weighter compatibility, but the value is L2. Documented in `trainer.py` module docstring.

3. **Backward order in meta+CL is load-bearing.** `loss_second.backward(retain_graph=True); optimizer2.step(); loss.backward(); optimizer.step()`. Swapping the order changes the gradient accumulated into `policy_net` (because `comparison_1` is produced from its layers). Preserved exactly; unit-tested via both-nets-move assertion.

4. **Paper target-sync frequency is 500 for CL, not 1000.** (CL paper line 1121 — differs from standard SARL.) Encoded in `config/training/sarl_cl.yaml`.

5. **CL uses Huber `cae_loss`, not BCE.** SARL+CL line 342 imports the SARL Huber variant (`maps.experiments.sarl.losses.cae_loss`), not the BCE variant in `maps.components.losses`.

## Deferred / punted items

- **`SARL_CL/examples_cl/AC_lambda.py` retention** — decision deferred to Sprint 07 (baseline-comparison scope lock). If Sprint 07 produces the AC-λ comparison chart via the paper's figures, this file can go; otherwise, it stays.
- **Full SARL multi-seed reproduction runs** — deferred to Sprint 07 on Narval. 04b proves architectural parity (the refactor is faithful); 07 will prove scientific parity (numbers match the paper's z-scores).
- **Curriculum-multi-game schema** — the current `run_training_cl` saves a simpler single-checkpoint format, not the paper's per-game optimizer/replay-buffer bookkeeping. Fine for single-stage runs and 2-stage curriculum tests; Sprint 06 can extend if needed for the exact paper curriculum sequence.

## Pointers

- Sprint 04b spec: `docs/sprints/sprint-04b-rl-split-and-legacy-delete.md`
- SARL ports: `src/maps/experiments/sarl/`
- SARL+CL ports: `src/maps/experiments/sarl_cl/`
- Configs: `config/training/sarl.yaml`, `config/training/sarl_cl.yaml`
- CLIs: `scripts/run_sarl.py`, `scripts/run_sarl_cl.py`
- Parity harness: `tests/parity/sarl/` (3 tiers + reference extracts)
- Integration smokes: `tests/integration/sarl/`, `tests/integration/sarl_cl/`

## Next up

Sprint 06 — Reproduction (perceptual): Blindsight + AGL multi-seed runs on Mac CPU, producing the paper's z-score tables for the perceptual-domain settings. No code changes; runs + reports only.
