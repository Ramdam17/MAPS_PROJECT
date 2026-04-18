# Sprint 06 — Report

**Period:** 2026-04-18 (single-day)
**Branch merged:** `repro/perceptual`
**Spec:** [`sprint-06-reproduction-perceptual.md`](../sprints/sprint-06-reproduction-perceptual.md)

## Outcome — shipped vs planned

| Task | Status | Notes |
|------|--------|-------|
| 6.1 Seed + setting matrix | ✅ | `--seeds` CLI override added to `run_blindsight.py` + `run_agl.py`; dry-run validated. |
| 6.2 Run the grid | ✅ | 40 Blindsight + 40 AGL cells at 200 epochs × 10 seeds × 4 settings, all on M-series CPU. |
| 6.3 Aggregation | ✅ | `scripts/aggregate_perceptual.py` emits `perceptual_summary.{json,md}`; exits non-zero on missing cells. |
| 6.4 Paper z-score comparison | ❌ (blocked) | See §Findings below — metric gap, not a run failure. |
| 6.6 AGL awareness split | ❌ (blocked) | Pre-existing gap flagged in spec; no path without porting eval from deleted monoliths. |

## Wall-clock (surprise)

| Grid | Cells | Wall-clock | Per-cell mean |
|------|------:|-----------:|--------------:|
| Blindsight (4 × 10 × 200 ep) | 40 | ~70 s | ~1.8 s |
| AGL (4 × 10 × 200 ep) | 40 | ~80 s | ~2.0 s |
| **Total** | **80** | **~2.5 min** | — |

Original estimate was ~20 h. Reality is ~800× faster because the refactored trainers are tight and the dimensions are small (Blindsight 100-d input, AGL 48-d input, 40-hidden). **No Narval needed for perceptual reproduction** — this fits on a laptop during a coffee break.

## Findings — what we have, what we don't

### What we have (run-level evidence)

`outputs/reports/perceptual_summary.md`:

| Domain | Setting | Mean loss | Std | Z vs baseline |
|--------|---------|----------:|----:|---------------:|
| Blindsight | neither | 4150.69 | 215.61 | 0.00 |
| Blindsight | cascade_only | 4085.09 | 220.64 | +0.30 |
| Blindsight | second_order_only | 4148.90 | 214.41 | +0.01 |
| Blindsight | both | 4084.13 | 221.12 | +0.31 |
| AGL | neither | 0.00 | 0.00 | 0.00 |
| AGL | cascade_only | 0.00 | 0.00 | n/a |
| AGL | second_order_only | 50.20 | 4.57 | n/a |
| AGL | both | 39.23 | 3.08 | n/a |

Observations:
- Blindsight: `cascade_only` and `both` show small but consistent improvement over `neither` (Δmean ≈ 65, ~0.30σ). `second_order_only` does not — cascade is the load-bearing component on the first-order loss axis.
- AGL: the natural comparison axis is `loss_2` (wagering loss), but `loss_2` is 0 when `second_order=False` → z-score undefined for those rows. `both` < `second_order_only` by ~11 points (cascade helps wager training).

### What we don't have (and why)

**The paper's headline numbers are accuracies, not training losses:**
- Blindsight: 0.97 detection accuracy (z=9.01 vs baseline)
- AGL-High: 0.66 classification accuracy (z=8.20)
- AGL-Low: 0.62 classification accuracy (z=15.70)

Our current trainers (`src/maps/experiments/{blindsight,agl}/trainer.py`) expose `pre_train()` only. **The evaluation-pass code that turns trained weights into held-out accuracy was in the monolithic scripts deleted during Sprint 04b §4.7 and was never ported.** That's the real blocker — logged as `RG-001` and `RG-002` in `docs/TODO.md` under a new "Reproduction gaps" section.

Implication: the 0.30σ Blindsight effect above is **on training loss, not on the paper's reported detection metric**. It is not directly comparable to the paper's z=9.01. We make no claim that we have reproduced the paper's perceptual numbers until the evaluation pass is ported.

## Addendum — Blindsight eval pass ported (RG-002 partially closed)

After the initial closeout, ported the paper's `testing()` routine into
`BlindsightTrainer.evaluate()` (see `feat/blindsight-evaluation` branch,
merged into Sprint 06 work). Config-first: the 200-trial test size and
per-condition wager thresholds (0.5 / 0.5 / 0.15) now live in
`config/env/blindsight.yaml`.

Re-ran the 40-cell grid with the eval step attached. Per-setting headline
(discrimination accuracy on superthreshold, 10 seeds each):

| Setting | Mean | Std | Z vs baseline | Paper z |
|---------|-----:|----:|---------------:|--------:|
| neither | 0.729 | 0.065 | +0.00 | 0.00 |
| cascade_only | 0.752 | 0.054 | +0.36 | n/a |
| second_order_only | 0.728 | 0.059 | −0.02 | n/a |
| both (full MAPS) | 0.755 | 0.054 | +0.40 | **9.01** |

The discrimination-accuracy effect is in the right direction (MAPS variants
beat the vanilla baseline on both mean and effect size) but the magnitude is
far below the paper's z=9.01. Two remaining hypotheses:

1. **Metric definition mismatch** — the paper's 0.97 may refer to *wager*
   (metacognitive) accuracy, not first-order discrimination. The reference
   `testing()` flattens a `(N, 2)` wager tensor against a `(N,)` target
   vector before thresholding, which is either a subtle protocol we need
   to mirror exactly or a reference-code quirk. Needs a close re-read of
   the paper's §3 methods.
2. **Training regime drift** — 200 epochs × 100 patterns/epoch may not
   drive the CAE loss to the same plateau as the reference runs. Needs
   a loss-trajectory diff against the original notebook's saved curves.

Neither is a protocol error in our code path; both are *reproduction-depth*
questions worth a dedicated follow-up, not a silent tweak. RG-002 is
therefore "eval path shipped, headline-number match deferred".

**Test coverage:** 6 new unit tests under
`tests/unit/experiments/blindsight/test_evaluate.py` cover the eval
contract (condition set, with/without second-order, threshold plumbing,
build precondition).

## Addendum 2 — AGL eval port + new gap RG-003

Ported AGL_TMLR.py `testing()` into `AGLTrainer.evaluate()` (same pattern
as RG-002): generates (grammar A + grammar B) held-out batch, cascades
through both networks, returns `classification_precision` and (when
second_order is enabled) `wager_accuracy`. The aggregation script now
performs the paper's post-hoc seed-pool High/Low awareness split.

Finding: `classification_precision` is ~0.08 across **all four settings**
for every seed (std ~0.03). Root cause is the reference behavior
`AGLTrainer.pre_train` inherits — the first-order network is reset to its
*initial* random weights at the end of pre_train (AGL_TMLR.py L751). So
post-pre_train, the first-order is untrained. The paper's 0.66 /
0.62 High/Low awareness numbers come from a **downstream training phase**
(supervised Grammar A vs B classification on the pre-trained 2nd-order,
with a fresh first-order) that lived in the deleted `AGL/AGL_TMLR.py` and
is **not yet ported**. Logged as **RG-003** in `docs/TODO.md`.

What's shipped in this addendum:
- `AGLTrainer.evaluate()` + 5 unit tests under `tests/unit/experiments/agl/`
- `config/training/agl.yaml` eval section (patterns_number, wager_threshold)
- `run_agl.py` writes eval metrics into `summary.json`
- `aggregate_perceptual.py` computes high/low awareness seed-pool split
- Fast-test suite: **206 passed** (+5 AGL eval + 6 Blindsight eval)

RG-001 "eval path" is closed; RG-003 is the new downstream-training blocker.

## Deviations logged

None — no hyperparameter changes from `config/training/{blindsight,agl}.yaml`. The gap is a missing evaluation pipeline, not a deviation.

## Pointers

- CLIs: `scripts/run_blindsight.py`, `scripts/run_agl.py` (now support `--seeds "42,43,..."`)
- Aggregation: `scripts/aggregate_perceptual.py`
- Raw outputs: `outputs/{blindsight,agl}/<setting>/seed-<NN>/summary.json`
- Report: `outputs/reports/perceptual_summary.{json,md}`
- Gap register: `docs/TODO.md` → "Reproduction gaps"

## Next up

Two viable tracks, independent:

1. **Port the paper's evaluation pass** (unblocks Sprint 06 headline numbers). Recover the eval code from `git log BLINDSIGHT/Blindsight_TMLR.py` / `AGL/AGL_TMLR.py` pre-SHA `d90d8f8` and port into the trainer modules. Small, bounded, pays off the entire perceptual z-score table.

2. **Proceed to Sprint 07** (SARL reproduction on Narval). SARL has its own eval flow (mean-return-over-final-100-episodes) already embedded in the training loop — no equivalent gap. Can run in parallel to (1).
