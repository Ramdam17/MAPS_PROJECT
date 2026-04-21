# Component review — `src/maps/experiments/agl/trainer.py`

**Sprint-08 D.26. Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/agl/trainer.py` (412 L).
**Paper sources :** MAPS §A.2 AGL, Table 10 (hyperparams), Table 5b/5c (High/Low awareness results).
**Student source :** `external/paper_reference/agl_tmlr.py` (2785 L).
**DoD :** read-only audit written. No code touched. Findings inform D.28 plan.

---

## (1) Structural finding — RG-003 root cause : missing training phase

The paper's AGL protocol is **3-phase per network** :

1. **Pre-training** on random-grammar words (`GrammarType.RANDOM`).
2. **Training** on Grammar A words — **12 epochs for "High Awareness" networks, 3 epochs for "Low Awareness"**.
3. **Testing** on mixed Grammar A + B.

Student `train()` at L1436 orchestrates this per seed :

- Builds 1 pair (`first_order_network`, `second_order_network`).
- Calls `pre_train()` (L1489) → the random-grammar phase.
- Calls `create_networks()` (L1495) → replicates the pair into **`num_networks=20` copies** (10 "high", 10 "low").
- Calls `training()` twice :
  - L1499 : `training(networks, num_training_high=12, 0, 10, ...)` → trains the first 10 copies for **12 epochs on Grammar A**.
  - L1504 : `training(networks_high, num_training_low=3, 10, 20, ...)` → trains the last 10 copies for **3 epochs on Grammar A**.
- Calls `testing()` on the combined 20 trained networks.

**Port status :**

- ✅ `pre_train()` present (matches student `pre_train` L619).
- ❌ **No `training()` method**. No 20-network duplication. No Grammar-A training phase.
- ⚠️ `evaluate()` claims *"the paper's High/Low Awareness split is a post-hoc seed-pool split"* (L343-346). **This is incorrect.** High vs Low is **not** a seed-pool split — it's 10 networks × 12 epochs vs 10 networks × 3 epochs, all within the same seed. Our port evaluates a single freshly-pretrained network.

**→ RG-003 structural cause confirmed (D-agl-training-missing in deviations.md).** Full fix requires porting `training()` + `create_networks()` + re-designing `evaluate()` around 20 networks per seed. Scope D.28.

---

## (2) Architectural bug inherited from student — wager head hidden layer dropped

Same bug pattern as Blindsight D.25 (RG-002) :

**Student `SecondOrderNetwork.__init__(use_gelu, hidden_second)`** at `agl_tmlr.py:211-256` :
- Accepts `hidden_second` argument (passed as **48** by `main()` L2564).
- **Never uses it** — body hardcodes `self.wager = nn.Linear(48, 1)` at L217.
- No hidden layer between comparator and wager — single linear + sigmoid.

**Paper §2.2** : *"as in Pasquali & Cleeremans (2010)"* — implies a hidden MLP layer per that citation.

**Port status :**
- Port's `SecondOrderNetwork` is shared with Blindsight (from `maps.components.second_order`).
- After D.25, it supports `hidden_dim` (default 0 = no hidden, bw-compat).
- **`config/training/agl.yaml` does NOT set `second_order.hidden_dim`** → currently 0 → port matches student bug → likely undertrains the wager head like Blindsight did.

**→ AGL fix candidate : add `second_order.hidden_dim: 48` to `config/training/agl.yaml`.** Pasquali 2010 hidden matches AGL input_dim. Same logic as Blindsight's `hidden_dim=100`. Low risk — can be ablated if bitwise parity matters for a test.

---

## (3) Config divergences — port vs student `main()` vs paper Table 10

Student `main()` at L2506-2628 runs all 6 factorial configs with **hardcoded args** :

```python
train(40, 48, 1, False, 1, 0.999, meta=..., cascade_rate=..., seeds=2, type_cascade=...)
```

Decoded : `hidden=40, hidden_second=48, factor=1, gelu=False, stepsize=1, gam=0.999, seeds_violin=2, optimizer='RANGERVA'` (default).

And `initialize_global()` L1648-1689 sets globals :
- `n_epochs_pre = 30` (comment says "default is 60" — disagrees with itself)
- `patterns_number_pre = 80`
- `learning_rate_1 = 0.4`, `learning_rate_2 = 0.1`
- `momentum = 0.5`, `temperature = 1.0`, `threshold = 0.5`, `bits_per_letter = 6`
- `num_networks = 20` (comment says "default is 30")
- `num_training_high = 12`, `num_training_low = 3`

| Knob | Paper T.10 | Student `main()` | Student `initialize_global()` | Port `config/training/agl.yaml` | Action |
|:--|:--:|:--:|:--:|:--:|:--|
| `first_order.hidden_dim` | 40 | 40 | — | 40 | ✅ aligned |
| `second_order.hidden_dim` | silent | **48** (dropped in code) | — | **0** (not set) | ❌ **add 48** (analogous to Blindsight D.25) |
| `optimizer.name` | RANGERVA | (default) | — | **ADAMAX** | ❌ **RangerVA** — needs `torch_optimizer` dep |
| `scheduler.step_size` | 1 | **1** | — | **25** | ❌ **align to 1** |
| `scheduler.gamma` | 0.999 | **0.999** | — | **0.98** | ❌ **align to 0.999** |
| `train.n_epochs` (pre) | 60 | — | **30** | **200** | ❌ **align to 60** (paper) or 30 (student actual). |
| `train.batch_size` (pre) | silent | — | **80** | **80** | ✅ aligned |
| `train.lr_first_order` | 0.4 | — | 0.4 | 0.4 | ✅ |
| `train.lr_second_order` | 0.1 | — | 0.1 | 0.1 | ✅ |
| `train.momentum` | silent | — | 0.5 | 0.5 | ✅ |
| `bits_per_letter` | 6 | — | 6 | 6 | ✅ |
| `num_networks` | silent | — | 20 (comment "default 30") | **N/A** (port missing) | ❌ D.28 — port creates 20 copies |
| `num_training_high` | 12 | — | 12 | **N/A** | ❌ D.28 |
| `num_training_low` | 3 | — | 3 | **N/A** | ❌ D.28 |

### Note on `n_epochs_pre`

Paper Table 10 says 60. Student `initialize_global` sets 30 with a comment *"default is 60"*. Self-contradictory. Cross-check with Blindsight (where paper T.9 said hidden=60 but student actual was 40) : **student code is authoritative for reproducing paper numbers**. Recommend **n_epochs_pre = 60** (paper) first, ablate to 30 if results don't match.

### Note on RangerVA

`RangerVA` is from the `torch-optimizer` package (not stdlib PyTorch). Adding it as a dependency is the only paper-faithful path. Alternatives :
- Use `RAdam` from `torch-optimizer` (close cousin, simpler).
- Keep ADAMAX and document the deviation (current behaviour).

---

## (4) Other `trainer.py` structural notes

### (4a) ✅ First-order weight reset after pre_train (L321)

Port `self.first_order.load_state_dict(self._initial_first_order_state)` at L321 correctly
replicates student L751. Comment at module docstring (L27-37) accurately documents the reference
behaviour and contrast with Blindsight (which does NOT reset). ✅ no issue.

### (4b) ✅ `target_second` computation (L288)

Port calls `target_second(batch.patterns, h2)` at L288, which (per `data.py` L421-441 in student)
computes `1` iff the top-k (winner-takes-all per 6-bit chunk) output positions **exactly match**
the k active input positions, else 0. This is per-batch element, not per-unit. Already reviewed
at data.py level (see `docs/reviews/agl-data.md` if it exists; else scheduled D.27).

### (4c) ⚠️ `wager.squeeze()` at L282, L403 — shape fragility

`wager = wager.squeeze()` — collapses all dim-1 axes. If `wager_units=2` (paper eq.3), shape is
`(B, 2)` which won't squeeze. The current path is safe for `n_wager_units=1` but the conditional
equivalent of Blindsight D.25's branching (`if int(self.cfg.second_order.n_wager_units) == 1 ...`)
is **missing here**. If we ever flip `n_wager_units=2` for AGL, loss/eval break.

**Recommendation :** plumb the same 1-vs-2-unit branch into AGL trainer as was done for
Blindsight in D.25. Low priority — deferrable until we actually want 2-unit for AGL.

### (4d) ✅ Factorial setting semantics (L76-92)

Mirror of `BlindsightSetting`. Same 4-cell abstraction. Works identically.

### (4e) ⚠️ Eval loop : classification_precision metric (L384-392)

Port computes precision per winner-takes-all chunk. Student `testing()` L1150+ computes more
metrics (`precision_2nd_order`, `recall_2nd_order`, `f1_2nd_order`, `accuracy_2nd_order`) —
port only returns 2 metrics. If the paper reports F1 or recall anywhere, we'd miss it.
**D.28 action :** align with student `testing()` outputs.

### (4f) ⚠️ Eval works on 1 network per seed, not 20 (L385 et al.)

Current `evaluate()` runs forward pass on a single `self.first_order` + `self.second_order`.
Student `testing()` iterates over 20 networks per seed and aggregates.
**D.28 action :** restructure eval around network list.

---

## (5) What D.26 did NOT touch

- No code changes.
- No tests modified.
- No config edits.

The file needs extensive surgery for RG-003 resolution (D.28) but D.26 only catalogues. 

---

## (6) Recommended D.28 plan (tentative)

Single commit sweep, executed per-item and validated :

1. **Config** alignment with student `main()` (paper-auth):
   - `second_order.hidden_dim: 48` (Pasquali 2010).
   - `scheduler.step_size: 1`, `gamma: 0.999`.
   - `train.n_epochs: 60` (paper) — fallback to 30 if parity fails.
   - `optimizer.name: ADAMAX` **for now** (RangerVA needs `torch-optimizer` dep decision).
2. **Implement** `AGLTrainer.training()` — port student L904-1035.
3. **Implement** `create_networks()` replication logic — 20 copies per seed.
4. **Refactor** `evaluate()` — accept a `list[(FirstOrder, SecondOrder)]`, iterate, aggregate per
   high/low consciousness tier (first 10 vs last 10).
5. **Add** `--n-networks` and `--training-epochs-high/low` CLI flags.
6. **Port** student `testing()` metrics : `precision_1st`, `precision_2nd`, `accuracy_2nd`,
   `recall_2nd`, `f1_2nd`.
7. **Parity tests** : bit-match student `train()` end-to-end on a small seeds=1, num_networks=2 run.
8. **500-seed validation** on DRAC.

---

## (7) Summary

| Category | Count | Notes |
|:--|:--:|:--|
| ✅ Already aligned | 7 | hidden_dim=40, batch_size=80, lr_first/second, momentum, threshold, bits_per_letter, cascade params |
| ❌ Config fix (easy) | 4 | second_order.hidden_dim, scheduler.step_size, scheduler.gamma, n_epochs (paper 60 or student 30) |
| ❌ Config fix (medium) | 1 | optimizer RangerVA (new dep) |
| ❌ Structural (D.28) | 3 | `training()` method, `create_networks()` replication, high/low 20-network split in eval |
| ⚠️ Shape fragility | 1 | 2-unit wager path missing branching (bw-compat with D.25 pattern) |
| ⚠️ Metric coverage | 1 | eval returns only 2 metrics, student 5 |

**Total port-vs-code deltas : 9 actionable items. Half are pure config. Half are structural.**

**D.26 closes on this document. No code touched. Next : D.27 data.py review, then D.28 RG-003 structural fix.**
