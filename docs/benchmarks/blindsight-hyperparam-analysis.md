# Blindsight hyperparameter benchmark — analysis & plan

**Status:** 🔵 draft for review (2026-05-25)
**Branch:** `bench/blindsight-hyperparam-sweep`
**Goal:** identify which hyperparameters, if any, close the systematic
~4 % gap between Juan's reference Blindsight code and paper Table 5a
values, via a paper-rigor factorial sweep (N = 500 seeds per cell).
**Required:** user validation of the sweep grid BEFORE any code runs.

## Why this benchmark exists

The Sprint 12 N = 500 reproduction found a consistent ~4 % under-shoot
on Setting 4 (paper MAPS) — discrim 0.936 vs paper 0.97, wager 0.808
vs paper 0.85. The Sprint 09 ↔ Sprint 12 cross-check (2 × 2 grid,
N = 20) showed Sprint 12 is *not* a regression vs Sprint 09 — both
land in the same range. The gap is intrinsic to Juan's code.

But we haven't yet asked **which hyperparameter, if any, closes the
remaining gap.** Probes so far :

- Pasquali hidden init (Sprint 09 `uniform(0, 0.1)` vs Sprint 12
  PyTorch default `uniform(-0.1, 0.1)`) — falsified, no improvement.
- Single-seed test — useless given seed-to-seed std ~0.04 ; we need
  paired N=500 means.

A rigorous answer requires : pick the hyperparameters most likely to
matter, sweep them on a factorial grid at the paper's N = 500 each
cell, and compare means with proper CI.

## Hyperparameter inventory & sweep candidacy

For every hyperparameter that touches the Blindsight pipeline, we
audit : what the paper says, what Juan's code does, what our current
config sets, and whether it's a plausible source of the gap.

Source : paper Appendix B.1 Table 9 (Blindsight, p. 28),
`docs/reproduction/paper_tables_extracted.md`, `config/maps.yaml`,
`config/domains/blindsight/training.yaml`,
`external/paper_reference/blindsight_tmlr.py`.

### Architecture

| Hyperparameter | Paper Table 9 | Juan code | Our config | Gap candidate? | Notes |
|---|---|---|---|---|---|
| `first_order.input_dim` | 100 | 100 | 100 | — | Fixed by task (autoencoder dim) |
| `first_order.output_dim` | 100 | 100 | 100 | — | Same as input |
| **`first_order.hidden_dim`** | **60** | 40 (`main()` default) | **40** (D12.3) | **⚠️ YES** | **Paper says 60, code uses 40.** RG-002 H5 chose 40 because it "works", but the paper is explicit. Single biggest paper-vs-code mismatch. |
| `first_order.encoder_dropout` | not stated | 0.1 | 0.1 | low | Paper silent ; code consistent |
| `first_order.weight_init_range` | not stated | uniform(-1, 1) | uniform(-1, 1) | low | Paper silent ; code consistent |
| `second_order.input_dim` | 100 | 100 | 100 | — | = first_order.input_dim |
| `second_order.dropout` | not stated | 0.5 | 0.5 | low | Paper silent ; code consistent |
| **`second_order.n_wager_units`** | **2** (eq.3 raw logits + BCE-with-logits eq.5) | 1 (sigmoid) | **1** (D-001) | **⚠️ YES** | **Paper says 2, code uses 1.** Mathematically near-equivalent for binary targets but the gradient path differs (sigmoid in forward vs sigmoid in loss). |
| **`second_order.hidden_dim`** (Pasquali) | implicit ("as in Pasquali & Cleeremans 2010" §2.2) | 0 (bug — student accepts but doesn't wire) | **100** (D.25 H10 restored) | partial | Already established 100 > 0. Worth checking intermediate value (e.g. 60). |
| `second_order.wager_weight_init_range` | not stated | uniform(0, 0.1) | uniform(0, 0.1) | low | Paper silent ; code consistent |

### Cascade

| Hyperparameter | Paper Table 9 | Juan code | Our config | Gap candidate? | Notes |
|---|---|---|---|---|---|
| `cascade.alpha` | 0.02 (= 1/50) | 0.02 | 0.02 | — | Locked, paper-faithful |
| `cascade.n_iterations` | 50 | 50 | 50 | — | Locked, paper-faithful |

### Optimizer

| Hyperparameter | Paper Table 9 | Juan code | Our config | Gap candidate? | Notes |
|---|---|---|---|---|---|
| `optimizer.name` | Adamax | ADAMAX | ADAMAX | low | Match. Adam vs Adamax could be a sanity sweep. |
| `optimizer.lr_first_order` | 0.5 | 0.5 | 0.5 | — | Match |
| `optimizer.lr_second_order` | 0.1 | 0.1 | 0.1 | — | Match |

### Scheduler

| Hyperparameter | Paper Table 9 | Juan code | Our config | Gap candidate? | Notes |
|---|---|---|---|---|---|
| `scheduler.name` | StepLR (implied : step + gamma) | StepLR | StepLR | — | Match |
| `scheduler.step_size` | 25 | 25 | 25 | — | Match |
| `scheduler.gamma` | 0.98 | 0.98 | 0.98 | low | Match ; decay rate could be a sanity sweep |

### Training

| Hyperparameter | Paper Table 9 | Juan code | Our config | Gap candidate? | Notes |
|---|---|---|---|---|---|
| `train.n_epochs` | 200 | 200 | 200 | low | Match ; Sprint 12.5 probe at 500 was slightly *worse*, but rigorous N=500 test could differ |
| `train.batch_size` | not stated explicitly (`patterns_number`) | 100 | 100 | low | Match Juan ; paper silent |
| `train.noise_level` | (per-condition `subthreshold` baseline 0.1) | 0.1 | 0.1 | low | Paper §3.2 + Appendix A.1 |
| `train.data_factor` | not stated | 1 | 1 | low | Multiplicative on patterns_number ; Juan always uses 1 |
| `train.momentum` | not stated | 0.9 (unused — ADAMAX has its own) | 0.9 | — | Dead param, ADAMAX manages its own momentum |
| `train.threshold` | (per-condition wager threshold 0.5 / 0.15) | 0.5 default | 0.5 | — | Matches eval logic |
| `pre_training_condition` | superthreshold | superthreshold | superthreshold | — | Match. Note : MAPS only trains on superthreshold, never sees subthreshold/low_vision at training |

### Loss

| Hyperparameter | Paper Table 9 | Juan code | Our config | Gap candidate? | Notes |
|---|---|---|---|---|---|
| `losses.cae_lambda` | not stated (`lam`) | 1e-4 | 1e-4 | low | Match Juan |
| `first_order_loss.kind` | SimCLR (paper §2.1 eq.4 prose) | CAE (D-002) | CAE | excluded | D-002 ; SimCLR exists as stub Sprint 11.D but augmentations require Sprint 12+ domain work |
| recon term in `cae_loss` | not stated | `bce_sum` | `bce_sum` | low | Match Juan |
| `h(1-h)` quirk on ReLU encoder | not stated | active (mathematically wrong on ReLU but preserved for parity) | active | — | D-001 ; can't change without breaking Juan-parity |

### Eval procedure

| Hyperparameter | Paper Table 9 | Juan code | Our config | Gap candidate? | Notes |
|---|---|---|---|---|---|
| `eval.patterns_number` | not stated (~200 in `testing()`) | 200 (100 noise + 100 stim) | 200 | low | Match |
| `eval.wager_thresholds.superthreshold` | (implied 0.5) | 0.5 | 0.5 | — | Match |
| `eval.wager_thresholds.subthreshold` | (implied 0.5) | 0.5 | 0.5 | — | Match |
| `eval.wager_thresholds.low_vision` | (implied 0.15) | 0.15 | 0.15 | — | Match |
| Network mode at eval | not stated | `.eval()` (dropout off) | `.eval()` | low | Match Juan |
| Cascade iters at eval | (assumed = train cascade_iters) | iters_1, iters_2 from setting | iters_1, iters_2 from setting | — | Match |

### Out of scope

| Hyperparameter | Reason |
|---|---|
| `seed` | We *sweep* seeds (N=500) for variance estimation, not as a knob. |
| `device` (cpu/mps/cuda) | MPS non-deterministic (verified). Locked to CPU for reproducibility. |
| Numerical precision (fp32/fp64) | Locked fp32 (PyTorch default). Could test fp64 in future, but not for closing the gap. |
| SimCLR loss | Excluded — augmentations are domain-specific (Sprint 12.D bit_flip only), needs separate post-Sprint 12 sprint. |

## Sweep proposal — three knobs, three levels, factorial

**Knobs (high-impact, identified above) :**

1. **`first_order.hidden_dim`** ∈ {**40** (current default = student `main()`), **60** (paper Table 9), 100 (legacy port)} — 3 levels.
2. **`second_order.hidden_dim`** ∈ {0 (no Pasquali = student literal), **60**, **100** (current default = D.25)} — 3 levels.
3. **`second_order.n_wager_units`** ∈ {**1** (sigmoid = current), **2** (raw logits = paper §2.1 eq.3)} — 2 levels.

**Locked at paper-faithful Juan-code values :** everything else
(optimizer ADAMAX, lr 0.5 / 0.1, scheduler StepLR(25, 0.98), 200
epochs, batch 100, CAE λ=1e-4, cascade α=0.02 × 50 iters, dropout
0.1/0.5, eval as student).

**Settings :** all 6 paper settings (S1-S6). Allows full factorial
pattern in paper Table 5a + our S6 extension.

**Seeds :** **N = 500** per cell (matches paper convention exactly).
Seed pool : 42 .. 541.

**Total cell count :** 3 × 3 × 2 = **18 hyperparameter cells × 6
settings × 500 seeds = 54,000 runs.**

**Compute estimate :** 6 settings × 500 seeds at one cell takes ~40
min on CPU (verified empirically from the Sprint 12.5 reproduction
run). 18 cells × 40 min = **12 hours wall-clock on CPU**. Fits the
user's 8-hour limit at the edge ; could shave by parallelising
multiple cells if Mac CPU permits, or by dropping one knob level if
needed.

## Alternative grids

If 12 h is too long, two trimmed grids :

**Grid A (small) — 2 × 2 × 2 × 6 settings × N=500 = 24,000 runs ≈
5 h** : drop the middle level on both hidden_dim knobs (test only
{paper, code} extremes).

| first_hidden | second_hidden | n_wager_units |
|---|---|---|
| 40 (code) | 0 (literal) | 1 (code) |
| 40 (code) | 0 (literal) | 2 (paper) |
| 40 (code) | 100 (D.25) | 1 (code) |
| 40 (code) | 100 (D.25) | 2 (paper) |
| 60 (paper) | 0 (literal) | 1 (code) |
| 60 (paper) | 0 (literal) | 2 (paper) |
| 60 (paper) | 100 (D.25) | 1 (code) |
| 60 (paper) | 100 (D.25) | 2 (paper) |

**Grid B (proposed default, 3×3×2)** : as in the table above (12 h).

**Grid C (extended) — adds gamma and optimizer ≈ 36 h** :
- + `scheduler.gamma` ∈ {0.95, **0.98**, 1.0 (no decay)}
- + `optimizer.name` ∈ {**ADAMAX**, Adam}
- 3 × 3 × 2 × 3 × 2 × 6 × 500 = 162,000 runs ≈ **36 hours**.
- Probably overkill ; sanity check only.

**My recommendation :** **Grid B (12 h)** — covers the 3 high-impact
knobs at meaningful intermediate levels, fits in one overnight run,
and gives a clean 2×3 cross-table (first_hidden × second_hidden) per
n_wager_units × per setting.

## Methodology details

### Run-time invariants

- Device : `cpu` (MPS non-deterministic ; verified earlier with seed
  42 giving wager 0.730 on MPS vs 0.820 on CPU).
- Seed : passed via `set_all_seeds(seed)` per run (random, numpy,
  torch, PYTHONHASHSEED, cuda, mps).
- Cascade : α = 0.02, iters = 50 per side when on (per setting flags).
- Eval : 200 patterns per condition (100 noise + 100 stimulus),
  `.eval()` mode (dropout off), threshold per-condition (0.5 super /
  sub, 0.15 low_vision).

### Output layout

```
outputs/blindsight-benchmark/
├── plan.yaml                              # snapshot of the grid
├── h1-<X>_h2-<Y>_nwu-<Z>/
│   ├── setting-1-baseline/
│   │   ├── seed-42/
│   │   │   ├── summary.json
│   │   │   ├── losses_1.npy
│   │   │   ├── losses_2.npy
│   │   │   ├── first_order.pt
│   │   │   └── second_order.pt
│   │   ├── seed-43/...
│   │   ├── ... (500 seeds)
│   │   └── _cell_aggregate.json
│   ├── setting-2-cascade-1st/...
│   ├── ... (6 settings)
│   └── _hp_aggregate.json
├── ... (18 cells)
└── benchmark_report.md                    # final paper-comparable analysis
```

Storing all `.pt` for 54k runs = ~2 GB (.pt files ~36 KB each).
Acceptable on local disk but heavy. We can drop them with a CLI flag
if disk is tight.

### Aggregation

For each `(cell × setting × condition)` :
- mean, std, min, max, median, P25/P75 across the 500 seeds
- 95% CI of the mean (= mean ± 1.96 × std/√500)
- Z-score vs paper Table 5a values (for superthreshold only)

For each `(cell × setting × condition × metric)` :
- compared across cells (within-setting hyperparameter ablation)

Final `benchmark_report.md` :
- Headline table : Setting 4 superthreshold across all 18 cells
- Factorial summary : best cell per metric per setting
- Comparison vs paper Table 5a
- Recommendation : which cell (if any) closes the 4% gap

### Implementation

- `scripts/benchmark_blindsight.py` — orchestrator that iterates over
  cells × settings × seeds, invokes the existing
  `maps.domains.blindsight.cli` via `subprocess` or imports it
  directly.
- `scripts/aggregate_benchmark.py` — reads all `summary.json` and
  produces per-cell aggregates + the final report.
- Single-tip checkpoint logic so we can resume on crash.

### Done when

- [ ] User validates the grid (Grid A, B, or C, or modifications)
- [ ] Benchmark orchestrator written + tested on 1 cell × 1 setting × 10 seeds
- [ ] Full sweep launched (with `nohup` / `tmux` for the long run)
- [ ] Aggregation produces a clean `benchmark_report.md` with paper-
  comparable headline table
- [ ] If we find a cell that closes the gap → flag the hyperparameter
  change for Natalie ; update `docs/reproduction/deviations.md`
- [ ] If no cell closes the gap → conclusion "the 4% gap is intrinsic
  to Juan's code, not addressable by the swept hyperparameters" ;
  reports this with confidence
- [ ] Commit benchmark code + the aggregate report (NOT the 54k
  `summary.json` files — those go to a tarball in parent dir like
  Sprint 12.5)

## Open questions for the user

1. **Grid choice** : A (5 h), B (12 h, my reco), or C (36 h) ?
2. **Should the sweep also include `train.n_epochs` ∈ {200, 500}** ?
   Sprint 12.5 8-seed probe suggested 500 was *slightly worse* but
   that was N=8, low confidence. Adding this doubles the grid.
3. **Should we also run AGL** ? Same architecture (FirstOrderMLP +
   SecondOrderNetwork), different domain, also paper N=500. AGL
   Sprint 13 isn't ported yet so this requires the Sprint 13
   port first.
4. **Storage** : keep all 54k `.pt` files (~2 GB) for further
   analysis, or drop ?

Pending user response on these, no code runs.
