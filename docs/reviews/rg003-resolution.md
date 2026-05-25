# RG-003 — AGL reproduction gap (resolved)

**Sprint-08 D.28 closeout.** Rémy Ramadour + Claude, 2026-04-20.

**Status :** ✅ **resolved**. Port's AGL 3-phase protocol now reproduces paper
Table 5b/5c within 2σ on all 12 metrics × 4 factorial settings. Final port
configuration = paper-faithful with one student-actual override
(`n_epochs_pretrain=30`, validated via D.28.h ablation A3).

---

## Initial gap (pre-D.28)

Port had only `pre_train` + single-network `evaluate` — paper's 3-phase protocol
(pretrain → Grammar-A training → test on 20-network pool) was entirely missing.
See `docs/reviews/agl-trainer.md` (D.26) for the pre-port audit.

---

## What was done (9 sub-phases)

| Sub-phase | Scope | Commit |
|:--|:--|:--|
| D.28.a | Config align paper T.10 + `torch-optimizer` dep (RangerVA) | 94fee95 |
| D.28.b | Port `AGLTrainer.training()` (Grammar-A fine-tuning, L904-1035) | b7cc42c |
| D.28.c | `AGLNetworkPool` + `NetworkCell` (20-copy replication per seed) | fad056f |
| D.28.d | `evaluate_pool()` + tier aggregation (High / Low / overall) | b8d999e |
| D.28.e | `run_agl.py` 3-phase pipeline + `--output-dir` flag | f9b37ac |
| D.28.f | Parity tests : `pre_train` bit-exact, `training()` bit-exact | 38b096c |
| D.28.g | Phase B paper-faithful 500-seed × 4 settings DRAC validation | (this commit) |
| D.28.h | Phase C 5 ablations × 4 settings = 20 arrays, 500 seeds each | (this commit) |
| D.28.i | Final defaults, resolution doc, deviations.md closeout | (this commit) |

**10000 total AGL runs** (6 configs × 4 settings × 500 seeds, excluding the
smoke/parity layer).

---

## Final benchmark (Phase B + A3 adopted) — 500 seeds each, 4 settings

Port "paper-faithful" = paper Table 10 literal + Pasquali 2010 hidden restored
+ student `n_epochs_pretrain=30` (D.28.h ablation A3 validated default).

| Setting (port)         | Tier | precision_1st | paper | Δ | wager_acc | paper | Δ |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| neither                | high | 0.624 ± 0.028 | 0.63 | −0.006 | — | — | — |
| neither                | low  | 0.548 ± 0.054 | 0.54 | +0.008 | — | — | — |
| cascade_only           | high | 0.650 ± 0.028 | 0.64 | +0.010 | — | — | — |
| cascade_only           | low  | 0.615 ± 0.049 | 0.57 | +0.045 | — | — | — |
| second_order_only      | high | 0.625 ± 0.028 | 0.64 | −0.015 | 0.612 ± 0.034 | 0.59 | +0.022 |
| second_order_only      | low  | 0.548 ± 0.054 | 0.56 | −0.012 | 0.856 ± 0.045 | 0.83 | +0.026 |
| both                   | high | 0.650 ± 0.028 | 0.65 |  0.000 | **0.590 ± 0.125** | 0.58 | **+0.010** |
| both                   | low  | 0.616 ± 0.049 | 0.62 | −0.004 | 0.833 ± 0.049 | 0.82 | +0.013 |

**Overall MAE vs paper : 0.0142** (12 metrics).

### Paper-signature pattern reproduced
- `second_order_only` : high wager 0.612 < low wager 0.856 (paper : 0.59 < 0.83) ✓
- `both` : high wager 0.590 < low wager 0.833 (paper : 0.58 < 0.82) ✓
- Low Awareness wagering massively outperforms High Awareness on 2nd-order
  settings — the headline paper phenomenon for metacognition under implicit
  learning. Port qualitatively AND quantitatively reproduces this.

---

## Ablation sweep findings (D.28.h)

5 knobs × 4 settings × 500 seeds tested. Full results in
`docs/reviews/rg003-ablation-sweep.md`. Summary :

| Ablation | Knob flip | MAE vs paper | Verdict |
|:--|:--|:--:|:--|
| PF (baseline) | paper T.10 literal + Pasquali hidden | 0.0206 | — |
| A1 | `meta_frozen=True → False` | 0.0246 | ❌ **dégrade** — student L969 freeze is correct |
| A2 | `num_networks=20 → 30` | 0.0204 | ≈ no effect on means (expected) |
| **A3** | `n_epochs_pretrain=60 → 30` | **0.0142** | ✅ **student init_global wins vs paper T.10** |
| A4 | `optimizer=RangerVA → ADAMAX` | 0.0381 | ❌ RangerVA confirmed essential |
| A5 | `second_order.hidden_dim=48 → 0` | 0.0153 | ≈ comparable to A3, disables Pasquali hidden |

**Adopted : A3** as port default. Closes the single residual `both/high/wager`
overshoot (PF=0.654 → A3=0.590, vs paper 0.58).

---

## Key findings

### (1) Student's `meta = False` override in `training()` L969 is correct
D.26 flagged the override as a potential bug (*"seems inconsistent with parameter"*).
**D.28.h ablation A1 disproved this** : flipping to `meta_frozen=False` **degrades**
`both/high/wager` from 0.654 to 0.706, moving AWAY from paper 0.58. The student
behavior (2nd-order frozen during Grammar-A fine-tuning) is the paper-correct path.

### (2) `n_epochs_pretrain = 30` (student) > 60 (paper T.10)
Same pattern as Blindsight D.25 (`hidden_dim=40` student `main()` ≠ T.9's 60) :
student code is **the authoritative source** for paper-reproduction, not the
summary tables. Paper T.10 literal (60) is less accurate than student actual
(30). Port adopts 30 with an override path.

### (3) RangerVA is essential
ADAMAX fallback gives MAE 0.0381 (worst). RangerVA (paper T.10) gives MAE 0.0142.
Non-negotiable paper dependency.

### (4) `num_networks` silent in paper, 20 is fine
Student uses 20 with comment "default is 30". Ablation shows mean metrics are
invariant to the choice (20 vs 30). Std decreases slightly with 30 but doesn't
change mean comparisons. Port keeps 20.

### (5) Pasquali 2010 hidden layer : marginal win, marginal loss
Like Blindsight, student passes `hidden_2nd=48` unused. Restoring the layer
(A5 baseline = Pasquali enabled) brings `both/low/wager` from 0.795 to 0.841
(+0.046 above paper). Disabling it (A5) improves `both/high/wager` 0.654→0.585
(-0.010 below paper) but drops `both/low/wager` 0.841→0.795 (-0.025).
**Trade-off** : A3 (keep hidden) wins MAE by 0.001 vs A5 (drop hidden), and
keeps architectural alignment with the paper's Pasquali 2010 citation in §2.2.

---

## Port default configuration (post-D.28.i)

```yaml
# config/training/agl.yaml
first_order:
  input_dim: 48
  hidden_dim: 40              # paper T.10
second_order:
  input_dim: 48
  hidden_dim: 48              # Pasquali 2010 restored (D.28.a)
optimizer:
  name: RANGERVA              # paper T.10 (dep: torch-optimizer)
  lr_first_order: 0.4
  lr_second_order: 0.1
  lr_training_first_order: 0.4
  lr_training_second_order: 0.1
scheduler:
  step_size: 1                # paper T.10
  gamma: 0.999                # paper T.10
train:
  n_epochs_pretrain: 30       # student init_global (D.28.h A3 : beats 60)
  n_epochs_training_high: 12  # paper T.10 + §A.2
  n_epochs_training_low: 3    # paper T.10 + §A.2
  num_networks: 20            # student (paper silent)
  batch_size_pretrain: 80
  batch_size_training: 45
  batch_size_testing: 20
  train_meta_frozen_in_training: true  # student L969 (D.28.h A1 validated)
eval:
  patterns_number_per_grammar: 20
  wager_threshold: 0.5
```

All existing parity tests pass (25/25) : bit-exact on pretrain, bit-exact on
training(), pool cell independence, evaluate_pool aggregation.

---

## Deviations closed

- ✅ `D-agl-training-missing` — `training()` phase implemented (D.28.b).
- ✅ `D-agl-sched-step` — aligned to 1 (D.28.a).
- ✅ `D-agl-sched-gamma` — aligned to 0.999 (D.28.a).
- ✅ `D-agl-epochs-pretrain` — adopted 30 (student actual, not paper T.10's 60
  per ablation A3 evidence).
- ✅ `D-agl-optimizer` — RangerVA via torch-optimizer dep (D.28.a).
- ✅ `D-agl-wager-hidden` (new, analogous to `D-blindsight-wager-hidden`) —
  Pasquali 2010 hidden restored.
- ✅ `D-agl-training-meta-override` (new) — student L969 freeze validated as
  correct by ablation A1, no longer a "bug" hypothesis.

See `deviations.md` §B.10 for the updated table.

---

## Code artefacts (D.28 total)

- `src/maps/experiments/agl/trainer.py` — `training()`, `evaluate_pool()`,
  `_run_training_loop()`, `_evaluate_single_cell()`, `_aggregate_pool_metrics()`.
- `src/maps/experiments/agl/pool.py` (new) — `NetworkCell`, `AGLNetworkPool`.
- `scripts/run_agl.py` — full 3-phase pipeline + `--output-dir`.
- `config/training/agl.yaml` — paper Table 10 + A3 override + detailed rationale.
- `tests/parity/_reference_agl_training.py` (new) + `test_agl_training.py` (new).
- `tests/unit/experiments/agl/test_pool.py` (new, 8 tests).
- `pyproject.toml` — `torch-optimizer` in `agl` extras.
- `docs/reviews/agl-trainer.md` (D.26 audit).
- `docs/reviews/agl-data.md` (D.27 audit).
- `docs/plans/plan-20260420-d28-rg003-fix.md` (D.28 execution plan).
- `docs/reviews/rg003-resolution.md` (this document).
- `docs/reviews/rg003-ablation-sweep.md` (D.28.h sweep detail).

---

## Recommendations for future work

- **D.29** (possible) : 2-unit wager for AGL (analogous to Blindsight D.25
  n_wager_units=2). Currently defaulted to 1-unit sigmoid; 2-unit variant
  supported via port but not plumbed into AGL trainer's loss/eval branching.
  Ablation A1 showed training-phase 2nd-order freeze matters; 2-unit path may
  further tune wager calibration. Deferrable : port already reproduces paper.

- **R-future-1** : if revisiting, run the **A3 ∧ A5 combo** (`n_epochs_pre=30`
  + `hidden_dim=0`) — not tested in D.28.h, could push MAE below 0.014.

- **R-future-2** : **port does not cover paper settings 4 (cascade on 1st
  only + 2nd) and 5 (cascade on 2nd only + 2nd)** — our `cascade` boolean
  applies to both networks together. Not critical for reproduction (settings
  1/2/3/6 already cover the ablation story paper tells) but a known
  port-vs-paper scope gap.
