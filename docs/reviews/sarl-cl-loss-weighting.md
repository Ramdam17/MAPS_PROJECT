# Component review — `src/maps/experiments/sarl_cl/loss_weighting.py`

**Review sub-phase :** Sprint-08 D.16.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl_cl/loss_weighting.py` (187 L, 3 helpers + 1 class).
**Paper sources :** §2.3 p. 9 (Continual Learning), eq.15-17 (running-max normalization).
**Student sources :** `external/paper_reference/sarl_cl_maps.py:510-607`.

**Callers du port :**
- `sarl_cl/trainer.py:65, 128, 129` (kwargs type annotation + usage).
- `sarl_cl/training_loop.py:60, 360, 361, 577-580` (instantiation per-network).

**DoD D.16** : doc créé, 4 symbols audités, 3 dead-code helpers flaggés, 0 code touché.

---

## (a) `DynamicLossWeighter` class — le cœur

Port L105-187 vs student L535-607.

### Math paper eq.15-17

Paper eq.15 verbatim (paper_equations_extracted.md:305) :
$$\mathcal{L}_{\mathrm{reg}} = \frac{1}{\max_t(\mathcal{L}_{\mathrm{reg}}(t))} \sum_k \| \theta_k - \theta^{\mathrm{teacher}}_k \|_2^2$$

Donc la loss normalisée = **raw / running_max**. Formule symétrique pour eq.16 (feature) et eq.17 (task).

### Mapping port ↔ student

| Aspect                     | Student                                         | Port                                              | Match |
|:---------------------------|:------------------------------------------------|:--------------------------------------------------|:-----:|
| `__init__` state           | 3 dicts (moving_avgs=1.0, historical_max=-inf, historical_max_prev=-inf), steps=0, update_interval=10_000, scale_factors={k: 1.0} | idem L139-152 | ✅ |
| `update(losses)`           | detach tensors, update moving_avgs (most recent), snapshot historical_max_prev at mid-interval, update historical_max=max(...) | idem L156-171 (plus explicit `value.item()` cast) | ✅ |
| `weight_losses(losses)`    | `value / (historical_max[k] + epsilon)` per key, epsilon=1e-16 | idem L177-179 | ✅ |
| `get_stats()`              | returns copies of moving_averages / historical_max / scale_factors | idem L181-187 | ✅ |

✅ **Bit-parity student** sur les 4 méthodes + state.

### (a1) Running max persistence semantic

Le weighter est instancié **une fois par network** dans `training_loop.py:577-580` (un pour 1st-order,
un pour 2nd-order si meta). Il **persiste across curriculum games** — pas de reset au game boundary.

Paper §2.3 p.9 ne spécifie pas explicitement si le max doit persister cross-game ou reset. Student
préserve → port match. ✅

### (a2) Dead path `historical_max_prev`

Student L549-553 + L571-572 snapshotent `historical_max_prev` au milieu de l'update_interval, mais
le code qui **utilise** `historical_max_prev` (L574-577) est commenté. Résultat : c'est du **dead
state** — écrit mais jamais lu.

Port L149 + L168-169 préserve ce dead path pour parity. ✅ **Cohérent** avec policy "preserve
paper" (ça sert aussi au checkpoint D.13 comme debug artefact).

### (a3) Dead path `scale_factors`

Student L557 + L581-585 (commented) — `scale_factors` aussi initialisé à 1.0 puis commented-out.
Port L152 + L186 (dans `get_stats`). Dead but kept parity.

### (a4) `update_interval = 10_000` + comment mensonger

Student L556 dit en commentaire *"Update historical max every 100 steps"* mais la valeur est
`10_000`. **Student comment misleading**. Port L132-133 supprime le commentaire erroné et garde
juste la valeur. ✅ **Port clarification safe** (pas paper-divergence).

### (a5) `epsilon = 1e-16` stability

Port `weight_losses(..., epsilon=1e-16)` — même valeur hardcodée student L589. Protège contre
`historical_max[k] = 0.0` pour les losses qui démarrent à 0. ✅

## (b) 3 helpers standalone — audit dead-code

### `update_moving_average(current_avg, new_value, momentum=0.9)` L48

**EMA formula** : `new_avg = momentum·current_avg + (1-momentum)·new_value`.

**Student usage** : student L531-533 définit la fonction. `grep` student : **jamais appelée** (remplacée par
running-max dans le weighter). Abandoned approach.

**Port usage** : `grep update_moving_average src/` → **0 caller en prod** hors self-re-export
`__init__.py`.

→ **Dead code** préservé pour parity + future opt-in (docstring L56-58 documente l'intention).

### `min_max_norm(mean, values)` L63

**Math** : `return (mean / max(values), max(values))`.

**Port docstring L68-70** : *"Does **not** compute a true min-max normalization despite its name;
this is a paper-code quirk preserved verbatim."* ✅ Quirk flaggé.

**Callers** : `grep` → **0 caller** en port src/. Student caller non-trivial à identifier — utilisé
probablement dans analysis scripts student non-portés.

→ **Dead code in src/** préservé.

### `individual_losses(output, target, loss_fn)` L77

**Math** : per-sample loss → list → 95e percentile.

**Callers** : `grep` → **0 caller** en port src/. Student commenté L517-528 dit *"Used by some
exploratory analyses"*.

→ **Dead code** préservé.

## (c) Checkpoint compatibility (D.13 cross-ref)

`_persist_checkpoint_cl` sauvegarde `loss_weighter` + `loss_weighter_second` objets **complets**
via `torch.save(weights_only=False)`. Confirmé pour D.13 : le `DynamicLossWeighter` est picklable
(state = 5 dicts + 2 ints + 1 tuple). ✅

## (d) Public API — `__init__.py` exposure

Port `sarl_cl/__init__.py:40-43, 67-80` expose les 4 symbols (DynamicLossWeighter +
update_moving_average + min_max_norm + individual_losses).

**Piste D16-fix-1** (skip) : demote les 3 dead-code helpers à privés (`_update_moving_average`,
etc.) pour signaler leur statut. Éviterait l'implication qu'ils sont stables API. **Skip** pour
parity + user policy "preserve paper-reproduction code".

## Cross-reference deviations.md

- **D-sarl_cl-lossweight-normalization** (⚠️ info, existante) : *"eq. 15-17 `1/max_t(L(t))`
  inline running max → `DynamicLossWeighter` class"* — status keep (equivalent).
  ✅ **Confirmed D.16**: port = student bit-parity, math paper eq.15.

### Aucune nouvelle deviation surfacée.

## Fixes identifiées D.16

| ID        | Fix                                                                  | Scope                           | Effort |
|:----------|----------------------------------------------------------------------|---------------------------------|:------:|
| D16-fix-1 | (skip) demote 3 dead helpers à `_private` names                      | `__init__.py` exports           | skip   |
| D16-fix-2 | (skip) supprimer `historical_max_prev` + `scale_factors` dead state  | `DynamicLossWeighter` body      | skip   |

Tous skip : parity + "preserve paper-reproduction code" policy.

## Résumé — `loss_weighting.py`

- ✅ **`DynamicLossWeighter` math bit-parity student** sur 4 méthodes + state init.
- ✅ **Paper eq.15-17 respectée** via `weight_losses = raw / (historical_max + ε)`.
- ✅ **Port documentation gains** : docstrings détaillées, comment mensonger student
  supprimé, dead paths explicitement commentés, epsilon hardcoded à 1e-16 documenté.
- ⚠️ **3 dead helpers** (`update_moving_average`, `min_max_norm`, `individual_losses`) :
  0 caller prod, préservés parity. DETTE candidate ? Skip (policy preserve).
- ⚠️ **Dead state** `historical_max_prev` + `scale_factors` : écrits jamais lus (commented code
  in student). Parity preserved.
- ✅ **D.13 checkpoint compat** : weighters round-trip via torch.save(weights_only=False).
- ✅ **Persistence cross-game** : même weighter across curriculum, student match, paper silent.
- **0 divergence paper, 0 bug, 0 new deviation.**

**D.16 clôturée. 0 code touché, 2 skips dead-code.**
