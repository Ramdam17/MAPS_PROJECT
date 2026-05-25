# Sprint-08 Phase C — closeout report (component reviews)

**Dates :** 2026-04-19 → 2026-04-20.
**Scope :** review + fix of `src/maps/components/` + `src/maps/networks/` + `src/maps/utils/`
vs paper + student reference code.
**Reviewer :** Rémy Ramadour + Claude.
**Deliverables produced :** 7 review docs + 1 closeout report.

---

## Sub-phase roster (17 total)

| #   | Title                                              | Status | Deliverable / key output                                        |
|:---:|----------------------------------------------------|:------:|:----------------------------------------------------------------|
| C.1 | Review `components/cascade.py`                     | ✅     | `docs/reviews/cascade.md` — 🚨 cascade=no-op on deterministic path |
| C.2 | Apply C.1 fixes                                    | ✅     | α→cascade_rate rename; 2 analytical tests                       |
| C.3 | Review `components.ComparatorMatrix`               | ✅     | `docs/reviews/second_order.md §C.3` — bit-exact vs eq.1         |
| C.4 | Review `components.WageringHead`                   | ✅     | §C.4 — 🚨 softmax bug on n_wager_units=2                        |
| C.5 | Review `components.SecondOrderNetwork` composition | ✅     | §C.5 — dropout→cascade resolves C.1 paradox on 2nd-order path  |
| C.6 | Apply C.3/C.4/C.5 fixes                            | ✅     | Shape assert, raw-logit wager, DEFAULT_WAGER_INIT_RANGE, DETTE-1 |
| C.7 | Review `components.cae_loss` + D-002 deep dive     | ✅     | `docs/reviews/losses.md §C.7` — D-002 real SimCLR vs CAE divergence |
| C.8 | Review `wagering_bce_loss` + `distillation_loss`   | ✅     | §C.8 — pos_weight API bug, D-003 T² confirmed                   |
| C.9 | Review `weight_regularization` + callers audit     | ✅     | §C.9 — distillation_loss dead code (DETTE-3), 6 callers audited |
| C.10| Apply C.7/C.8/C.9 fixes                            | ✅     | 10 fixes; pos_weight dropped; DETTE-2/3 added; Phase D.22b sub-phase added |
| C.11| Review `networks.FirstOrderMLP`                    | ✅     | `docs/reviews/first_order_mlp.md` — 🚨 C.10 erratum (h post-ReLU) ; RG-002 hidden_dim confirmed |
| C.12| Apply C.11 fixes                                   | ✅     | cae_loss docstring hotfix; FirstOrderMLP.hidden_dim clarif      |
| C.13| Review `utils/` (seeding, config, paths, logging)  | ✅     | `docs/reviews/utils.md` — 0 divergence paper, 3 docstring gaps  |
| C.14| Review `utils.energy_tracker`                      | ✅     | `docs/reviews/energy_tracker.md` — dead-code-in-src, DETTE-4    |
| C.15| Apply C.13/C.14 fixes                              | ✅     | matplotlib lazy; training_loop seed comments; DETTE-4 entry     |
| C.16| Audit public API (`__init__.py` × 3)               | ✅     | `docs/reviews/public_api.md` — all exports correctly wired       |
| C.17| Apply C.16 fix + Phase C closeout                  | ✅     | Import harmonisation + this report                              |

---

## Commits Phase C (17 commits)

```
6e3f00a  docs(review): review src/maps/components/cascade.py (Phase C.1)
8e841f0  test(cascade): add analytical no-op tests for deterministic path (C.2 / C1-fix-1)
1e4edd0  refactor(cascade): rename alpha → cascade_rate (C.2 / C1-fix-3)
71682b2  docs(review): review components.ComparatorMatrix vs paper eq.1 (Phase C.3)
7b62173  docs(review): review components.WageringHead vs paper eq.2-3 (Phase C.4)
37c5ee3  docs(review): review components.SecondOrderNetwork composition (Phase C.5)
31bf018  fix(components): apply C.3/C.4/C.5 review fixes (shape assert, raw-logit wager, DETTE-1)
fd61f27  docs(review): review components.cae_loss + D-002 SimCLR-vs-CAE (Phase C.7)
9880b78  docs(review): review wagering_bce_loss + distillation_loss (Phase C.8)
a9b74af  docs(review): review weight_regularization + losses.py callers (Phase C.9)
9d81f05  fix(losses): apply C.7/C.8/C.9 fixes (docs, pos_weight removal, DETTE-2/3, D.22b)
e2eff9a  docs(review): review networks.FirstOrderMLP — flags cae_loss erratum (Phase C.11)
c2252ed  fix(docs): apply C.11 fixes — correct cae_loss h post-ReLU quirk (Phase C.12)
2372f0e  docs(review): review utils/ (seeding, config, paths, logging) (Phase C.13)
87b7720  docs(review): review utils.energy_tracker — DETTE-4 (Phase C.14)
4f7e7d6  fix(utils): apply C.13/C.14 fixes — docstring + matplotlib lazy + DETTE-4 (Phase C.15)
85c4465  docs(review): audit public API exports __init__ files (Phase C.16)
```
(C.17 closing commit added after this report.)

---

## Review docs produced

7 review docs totalling ~1700 lines of analysis :

| Doc                                        | Lines | Coverage                                    |
|:-------------------------------------------|:-----:|:--------------------------------------------|
| `docs/reviews/cascade.md`                  | ~290  | cascade primitive + 🚨 no-op finding        |
| `docs/reviews/second_order.md`             | ~510  | ComparatorMatrix + WageringHead + SecondOrderNetwork |
| `docs/reviews/losses.md`                   | ~710  | cae_loss + wagering_bce_loss + distillation_loss + weight_regularization |
| `docs/reviews/first_order_mlp.md`          | ~235  | FirstOrderMLP + make_chunked_sigmoid        |
| `docs/reviews/utils.md`                    | ~310  | seeding + config + paths + logging_setup    |
| `docs/reviews/energy_tracker.md`           | ~185  | NvidiaEnergyTracker + MLModelEnergyEfficiency |
| `docs/reviews/public_api.md`               | ~180  | 3 __init__.py audit                         |

---

## Critical findings consolidés

### 🚨 Paper-vs-code divergences (structurelles)

| ID    | Surface                   | Divergence                                                   | Status                     |
|:------|:--------------------------|:-------------------------------------------------------------|:---------------------------|
| D-001 | Wagering head             | Paper eq.3 = 2-unit raw logits (BCE-with-logits per unit) ; student/port Blindsight+AGL = 1-unit sigmoid | Port supports both paths after C.6; default=1 for parity. Paper variant now returns raw logits (bug fixed C.6). |
| D-002 | 1st-order loss            | Paper eq.4 = SimCLR/NT-Xent (Chen 2020) ; student/port = CAE (Rifai 2011) — mathematically distinct | Blocking Phase D.22b decision (impl SimCLR vs keep CAE). |
| D-003 | Distillation KL           | Hinton T² scaling absent in student+port                     | Confirmed, status quo OK (matches student). |
| D-004 | AGL decoder               | Chunked sigmoid per 6-bit letter (paper + student + port)    | Correctly handled via `make_chunked_sigmoid`. |

### 🚨 Findings découverts en Phase C

| Finding | Surface | Description | Resolution |
|:--------|:--------|:------------|:-----------|
| C1-F1 | cascade | Cascade = mathematical no-op on deterministic forward path (all iterations converge to h_raw in 1 step) | Partially resolved in C.5: 2nd-order path has dropout → cascade averages masks → non-trivial effect. SARL 1st-order path still no-op (flagged for Phase D). |
| C4-F1 | WageringHead | `n_wager_units=2` applied softmax but paper expects raw logits | Fixed C.6 (removed softmax). |
| C11-F1 | cae_loss docstring | C.10 docstring claimed `h` must be post-sigmoid; actually post-ReLU in all 3 domains | Fixed C.12. |
| C11-F2 | Blindsight config | `hidden_dim=100` diverges from paper Table 9 (`60`) — RG-002 H1 | Fix queued Phase D.25. |

### 🚨 Student code is inconsistent with paper (already documented, re-confirmed)

15 🆘 paper-vs-student divergences across SARL/Blindsight/AGL/MARL tables (Phase B audit). Student
code as vendored cannot produce Tables 5/6/7 z-scores. Phase C re-confirmed these on the components
we own; no new 🆘 surfaced.

---

## DETTEs opened (technical debt tracking)

| DETTE   | Surface                                                 | Reason kept                                                    |
|:--------|:--------------------------------------------------------|:---------------------------------------------------------------|
| DETTE-1 | `SecondOrderNetwork` + `SarlSecondOrderNetwork` doublon | SARL tied-weight architecture differs from Blindsight/AGL      |
| DETTE-2 | `components.cae_loss` + `sarl.losses.cae_loss` doublon  | BCE/MSE vs Huber, sigmoid vs ReLU — paper-faithful per-domain  |
| DETTE-3 | `distillation_loss` public but dead                     | Preserves student parity (class defined but never called paper-side) |
| DETTE-4 | `energy_tracker.py` dead in src/ (MARL consumer)        | Keep for MARL port; Phase H relocation                         |

All 4 DETTEs documented in `docs/reproduction/deviations.md`, resolution path = post Phase F.

---

## Fixes applied (25+ across 5 batch commits)

### C.2 — cascade fixes (2)
- α → cascade_rate rename (eliminate collision with EMA wagering α).
- 2 analytical no-op tests for deterministic path.

### C.6 — components batch (5)
- ComparatorMatrix shape assertion.
- WageringHead raw-logit path (remove erroneous softmax).
- `DEFAULT_WAGER_INIT_RANGE` module constant.
- 3 docstring updates.

### C.10 — losses batch (10)
- cae_loss docstring with D-002 note, post-sigmoid/post-ReLU clarif (corrected C.12).
- `wagering_bce_loss.pos_weight` parameter removed (was misnamed, 0 callers).
- `distillation_loss` dead-code inline comment.
- `weight_regularization` docstring: strict=True + teacher-frozen contract.
- `sarl/losses.py` state_dict docstring correction + ReLU quirk note.
- `deviations.md`: D-002 updated, DETTE-2 + DETTE-3 entries.
- Plan sub-phase D.22b added (SimCLR vs CAE decision).

### C.12 — first_order_mlp fixes (3)
- `cae_loss` docstring erratum correction (h post-ReLU universal).
- `make_chunked_sigmoid` clone-vs-in-place note.
- `FirstOrderMLP.hidden_dim` AGL-default clarif + RG-002 pointer.

### C.15 — utils batch (5)
- `set_all_seeds` docstring: `np.random.default_rng` gap noted.
- `config.CONFIG_ROOT` import-time warning.
- `energy_tracker` matplotlib lazy import (~300ms cold saved).
- `training_loop` (SARL + SARL+CL): inline comments on local seed re-seeding.
- `deviations.md`: DETTE-4 entry.

### C.17 — public API harmonisation (1)
- `agl/trainer.py` + `sarl_cl/trainer.py`: 2 imports harmonised to public API.

---

## Phase D — handoff

**Phase D scope (30 sub-phases)** : per-experiment reviews + fixes (SARL D.1-D.15, SARL+CL D.16-D.22,
+**D.22b new blocking sub-phase** for D-002 decision, Blindsight D.23-D.25, AGL D.26-D.28, final
D.29-D.30).

**Blocking dependencies** :
- **D.22b** (SimCLR vs CAE decision) blocks D.23 (Blindsight trainer) + D.26 (AGL trainer).
- **RG-002 fix** (D.25) depends on **D.23** (Blindsight trainer review for metric-mismatch diagnosis).
- **RG-003 fix** (D.28) depends on **D.26** (AGL trainer review for downstream training port).

**Recommended Phase D starting point** : **D.1** (SARL data review, no dependencies), then continue
sequentially respecting the no-parallel-flanks discipline.

**Global state at Phase C closeout** :
- Components layer : **reviewed + fixed, paper-faithful on structural points**.
- Networks layer : **reviewed + documented**.
- Utils layer : **reviewed + documented**, no paper divergences.
- Public API : **cohesive, no fantom names, no orphan exports**.
- `deviations.md` : **4 D-NNN + 4 DETTE-N + 3 G-NN** (Phase B + C merged).
- Plan `docs/plans/plan-20260419-review-and-reproduce.md` : Phase C section updated ✅.

---

## Methodology note — Orient→Do→Verify→Report→Commit→Wait

Strictly applied on all 17 sub-phases after the early-B rushed-execution rollback. No parallel
flanks. Every sub-phase gated on Rémy's explicit "go" between Orient and Do, and between Report
and Commit. 17/17 micro-sprints completed without re-work — the discipline paid off.

**Phase C clôturée 2026-04-20.**
