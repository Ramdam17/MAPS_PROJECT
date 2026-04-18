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
