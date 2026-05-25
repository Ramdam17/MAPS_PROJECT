# Blindsight `evaluate()` — line-by-line audit vs student `testing()`

**Sprint-08 D.25 — RG-002 H4 audit (read-only).**
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**Goal :** before changing any code, understand whether port `evaluate()` diverges
from student `testing()` — and if so, where.
**DoD :** written verdict, no code touched. Output hypothesis list for sbatch tests.

---

## (1) Metric computation — port vs student

### (1a) Discrimination accuracy

| Step | Port `trainer.py:422-430` | Student `blindsight_tmlr.py:866-871` |
|:--|:--|:--|
| delta | `n_eval // 2` = 100 | `int(100 * factor)` = 100 @ factor=1 |
| pred idx | `h2[delta:].argmax(dim=1)` | `output_first_order[delta:].argmax(dim=1)` |
| true idx | `batch.patterns[delta:].argmax(dim=1)` | `input_data[delta:].argmax(dim=1)` |
| aggregate | `(pred == true).float().mean()` | same, `round(..., 2)` |

✅ **Bit-match** modulo student's per-seed rounding to 2 decimals (purely cosmetic).

### (1b) Wager accuracy

| Step | Port `trainer.py:435-447` | Student `blindsight_tmlr.py:879-905` |
|:--|:--|:--|
| slice | `wager[delta:, 0]` (high-wager col) | `output_second_order[delta:].flatten()` (N×1 → N) |
| target | `batch.order_2_target[delta:, 0]` (paper: 1 if above thresh) | `argmax(stim_present[delta:]) > 0` (1 iff above thresh AND stim_idx≠0) |
| threshold | per-cond : 0.5 / 0.5 / 0.15 | same |
| metric | `(pred_bin == tgt_bin).mean()` | `(TP+TN)/(TP+TN+FP+FN)` — identical formula |

✅ **Bit-match** modulo a 1% edge case : student's `(argmax > 0)` convention
mis-labels stim_idx=0 trials as low-wager even when above threshold. Port uses
`order_2_target[:, 0]` directly (the paper-faithful semantic: 1 iff above thresh).
Impact : port slightly more generous on that ~1% slice; noise in error budget.

**Student bug preserved only in `data.py`'s RNG order, not in the target
semantics.** No port divergence detected.

---

## (2) Where do paper Table 5a numbers actually come from?

Paper Table 5a "MAPS, Blindsight suprathreshold" : discrim `0.97±0.02`, wager `0.85±0.04`.

### Student's paper-producing function is `main()` at L2164, NOT the grid search.

Student `main()` body (L2212–2310) runs the **6 factorial configurations**
back-to-back, each with the following **literal hardcoded arguments** :

```python
train(
    hidden=40,            # ← NOT 60 as in Table 9 !
    hidden_2nd=100,
    factor=1,             # ← data_factor=1 → patterns_number = 200 * 1 = 200
    stepsize=25, gam=0.98,
    meta=..., optimizer='ADAMAX',
    seeds=seeds_violin,   # = 10
    noise_level=0.0012,   # ← condition 0 (superthresh) ignores it (baseline=0)
    type_cascade=...,
)
```

**THIS is the code path that produces the paper's Table 5a numbers.** The grid
search at L2346 (`primary_scaling_factors=[0.0625, 0.125, 0.25, 0.5]`) is a
**separate scaling experiment**, not the headline result.

### 🚨 Discrepancy : Table 9 ≠ student main() ≠ port

| Knob | Paper Table 9 | Student `main()` | Port `config/training/blindsight.yaml` (post-D.25) |
|:--|:--|:--|:--|
| `hidden_dim` | **60** | **40** | **60** (just aligned to T.9 in D.25) |
| patterns/epoch | silent (T.9 silent on batch) | **200** (factor=1) | **100** (`train.batch_size`) |
| `n_epochs` | 200 | 200 | 200 |
| `lr_1 / lr_2` | 0.5 / 0.1 | 0.5 / 0.1 | 0.5 / 0.1 |
| `step_size / gamma` | 25 / 0.98 | 25 / 0.98 | 25 / 0.98 |

**The student's own code disagrees with Table 9 on `hidden_dim`.** Paper reports
`hidden=60` in the appendix table but trains+tests with `hidden=40` in `main()`.

**Port is doubly off vs the code-producing-paper-numbers :**
1. `hidden_dim=60` ≠ student main's 40 (we aligned to T.9, student used 40).
2. `batch_size=100` = half of student main's `patterns_number=200`.

---

## (3) Conclusions

1. **Metric computation is bit-faithful.** Port `evaluate()` matches student
   `testing()` to ~1% edge-case noise. H4 (metric mismatch) is **negated**.
2. **Port hyperparameters diverge from the code that produced paper Table 5a.**
   Two actionable config gaps :
   - **H5** : `first_order.hidden_dim = 40` (student `main()` value, **not** T.9).
   - **H6** : `train.batch_size = 200` (student `patterns_number`, factor=1).
3. **Current port result (hidden=60, batch=100) ≈ 0.666 wager, 0.863 discrim.**
   These ARE reproducible outcomes of the current port config — they just don't
   correspond to the paper's own code settings.
4. **D-001 (1-unit sigmoid vs 2-logit softmax) unaffected by this audit.** Both
   student and port use 1-unit sigmoid. Paper §2.2 prose says 2 logits. Still
   pending, but likely not the RG-002 root cause since student matches port.

---

## (4) Proposed sbatch tests (next step)

Two single-variable sweeps, isolated output paths, 500 seeds each :

| ID | Knob | Command fragment | Expected cost |
|:--|:--|:--|:--|
| H5 | `hidden_dim=40` | `-o first_order.hidden_dim=40` | 500 seeds × ~8s ≈ 10 min wall |
| H6 | `batch_size=200` | `-o train.batch_size=200` | 500 seeds × ~14s ≈ 18 min wall |

If H5 or H6 alone lifts wager near 0.85, we have the root cause. If only the
combined H5 + H6 does, both are necessary. If neither, other hypotheses (D-001
2-logit architecture, loss formulation) become primary.

**Decision logged for Rémy** : we aligned `hidden_dim` to paper Table 9 (60) in
D.25 step 1 — but **student main() uses 40**. We need to decide whether to :
(a) revert `hidden_dim` to 40 for paper-code parity, OR
(b) keep 60 (Table 9-faithful) and log the T.9↔code discrepancy as a new
    deviation.

My recommendation : **test H5 (hidden=40) first** — if it reproduces 0.85, we
know where the paper numbers actually came from, and can document the T.9↔code
mismatch transparently in `deviations.md`.

---

## (5) What this audit did NOT touch

- No change to `trainer.py`, `data.py`, or any config YAML.
- No new sbatch jobs submitted yet (pending Rémy's go/no-go on H5 & H6).
- D-001 (1-unit vs 2-logit) deferred — not the RG-002 root cause given the
  port-student parity established above.
