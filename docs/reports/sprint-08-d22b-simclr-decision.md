# D.22b — D-002 decision log : SimCLR vs CAE first-order loss

**Status :** ✅ Decided 2026-04-20 — Option 2 (keep CAE).
**Decider :** Rémy Ramadour.
**Scope :** Blindsight, AGL, SARL first-order loss family (paper eq.4).
**Blocking :** D.23 Blindsight trainer review, D.26 AGL trainer review (unblocked by this decision).

---

## Context

Paper §2.2 + eq.4 describes the first-order task loss as a **contrastive loss** with the
SimCLR / NT-Xent formula (Chen et al. 2020, *A Simple Framework for Contrastive Learning of
Visual Representations*, arXiv:2002.05709) :

$$
\mathcal{L}_{\mathrm{contrastive}} = \ell_{i,j}
= -\log \frac{\exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}
              {\sum_{k=1}^{2N} \mathbf{1}_{[k \ne i]} \, \exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
$$

The reference student code (`external/paper_reference/blindsight_tmlr.py`,
`agl_tmlr.py`, `sarl_maps.py`, `sarl_cl_maps.py`) **does not** implement SimCLR. It implements
a **Contractive AutoEncoder loss** (Rifai et al. 2011, *Contractive Auto-Encoders: Explicit
Invariance During Feature Extraction*, ICML 2011) :

$$
\mathcal{L}_{\mathrm{CAE}} = \underbrace{\mathrm{recon}(x, \hat{x})}_{\text{BCE or Huber}} + \lambda \cdot \underbrace{\|J_h(x)\|^2_F}_{\text{contractive penalty}}
$$

The student's docstring uses "contrastive" vocabulary (see e.g. `blindsight_tmlr.py:98-101` :
*"Contrastive loss plays a crucial role..."*) but the implementation is CAE. Naming collision,
not equivalence.

The two losses are **mathematically distinct** :

| Aspect        | SimCLR (paper eq.4)                        | CAE (student/port)                        |
|:--------------|:-------------------------------------------|:------------------------------------------|
| Objective     | Attract positive pairs / repel negatives   | Reconstruct + penalise encoder Jacobian   |
| Data contract | Positive pair `(z_i, z_j)` augmentations   | `(x, \hat{x})` input vs reconstruction    |
| Gradient via  | Normalised embeddings + softmax            | Reconstruction error + sigmoid derivative |
| Batch-coupled | Yes (negatives from batch)                 | No (per-sample)                           |
| Hyperparams   | `τ` (temperature)                          | `λ` (Jacobian weight)                     |

## Options considered

### Option 1 — Implement SimCLR paper-faithfully

**Cost** ~1-2 days engineering :
- New module `maps.components.losses.contrastive_simclr`.
- Data augmentation pipeline for positive pairs : non-trivial for Blindsight's binary patterns
  (what counts as a meaningful augmentation of a 100-dim noise vector ?) and AGL's discrete
  letter strings (meaningful grammar-preserving perturbations ?). SARL has an obvious candidate
  (crop / shift MinAtar frames) but paper silent on it.
- Temperature `τ` tuning required.
- All existing parity tests (`tests/parity/sarl/`, `tests/parity/sarl_cl/`) would need to be
  re-baselined or split into "student-path" vs "paper-path" tracks.

**Risk** : the paper's reported Tables 5/6/7 z-scores come from runs that used the student
code — which itself uses CAE. There is **no known SimCLR run** that produced the paper's
numbers. Implementing SimCLR moves us **further** from the empirical target, not closer.

### Option 2 — Keep CAE, document the divergence

**Cost** : ~0 (already in place post-D.15).

**Risk** : the Phase G reproduction report must openly state *"paper prose claims SimCLR but
all known implementations — including the paper's own code — use CAE ; our reproduction
matches the CAE path ; a future paper-faithful SimCLR run is an open question"*.

## Decision (2026-04-20, Rémy)

**Option 2**. Rationale :

1. The paper's **empirical claims** (Tables 5/6/7 z-scores) were produced by CAE-path code. To
   reproduce those numbers we must use CAE.
2. Option 1 would burn engineering time to **diverge from the empirical target** in pursuit of
   a prose claim that has never been validated against the paper's own results.
3. The divergence is surfaced honestly in `docs/reproduction/deviations.md` D-002 and in
   trainer docstrings, so no reader is misled.

## Future-proofing — config toggle

To keep Option 1 reachable without a code rewrite, the port exposes a config key :

```yaml
# config/maps.yaml
first_order_loss:
  kind: cae  # 'cae' (paper-faithful-via-student-code, current default)
             # 'simclr' (paper-faithful-via-eq4, not ported — reserved)
```

Each trainer (`sarl/trainer`, `blindsight/trainer`, `agl/trainer`) checks this at setup :

- `kind == "cae"` → current behaviour.
- `kind == "simclr"` → `NotImplementedError` pointing at this decision log.
- anything else → `ValueError`.

This gives a single well-typed place to flip if Phase F empirical results ever motivate a
SimCLR experiment. No additional plumbing needed post-flip beyond writing the SimCLR loss
function itself.

## Revisit conditions

Re-open this decision if **any** of the following holds :

1. **Phase F reproduction misses the paper's headline numbers** (Tables 5/6/7 z-scores) by a
   margin that can plausibly be attributed to the loss choice. Diagnostic : compare per-metric
   z-score deltas between our CAE runs and the paper's published values.
2. **A new empirical reference** appears (e.g. the original authors or a follow-up paper
   publishes a SimCLR-based MAPS run). Re-use their hyperparameters.
3. **An ablation request** comes from the PI (Guillaume) or a reviewer for a SimCLR comparison
   — implementation cost stays at ~1-2 days, the config toggle makes it a clean opt-in.

## Related artefacts

- `docs/reviews/losses.md §C.7 (a)(b)` — original C.7 deep-dive on the mathematical distinction.
- `docs/reproduction/deviations.md` D-002 — status row (updated to ✅ resolved with this decision).
- `docs/reproduction/paper_equations_extracted.md` eq.4 — verbatim paper formula.
- `config/maps.yaml` `first_order_loss` section — the toggle.
- 3 trainer guard-rails : `sarl/trainer.py`, `blindsight/trainer.py`, `agl/trainer.py`.
