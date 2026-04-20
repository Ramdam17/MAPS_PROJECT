# Component review — Public API exports (C.16)

**Review sub-phase :** Sprint-08 C.16 (`maps.components.__init__`, `maps.networks.__init__`,
`maps.__init__`).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**Files under review :**
- `src/maps/__init__.py` (7 L)
- `src/maps/components/__init__.py` (26 L)
- `src/maps/networks/__init__.py` (5 L)

**DoD C.16** : doc créé, audit exports vs callers, fixes listés, 0 code touché.

---

## `maps/__init__.py` — top-level

```python
"""MAPS — Metacognitive Architecture for Perceptual and Social Learning.
Core package. Submodules are populated in Phase 3 of the refactor
(see docs/sprints/sprint-00-remise-en-route.md).
"""
__version__ = "0.1.0"
```

- ✅ Pas de re-export des submodules (philosophie "explicit over implicit", cohérent CLAUDE.md).
- ✅ `__version__` présent.
- ⚠️ **Version `0.1.0` obsolète post Sprint-08** : aucun bump malgré 9+ commits feature/fix. Debate
  — skip en C.16.

## `maps.components/__init__.py` — re-exports composants

9 symbols exportés :

```python
__all__ = [
    "ComparatorMatrix", "SecondOrderNetwork", "WageringHead",           # second_order.py
    "cae_loss", "distillation_loss", "wagering_bce_loss",               # losses.py
    "weight_regularization",                                             # losses.py
    "cascade_update", "n_iterations_from_alpha",                         # cascade.py
]
```

- ✅ Ordre alphabétique.
- ✅ Tous les symbols `__all__` sont bien importés (pas de nom fantôme).
- ⚠️ **`DEFAULT_WAGER_INIT_RANGE`** (ajouté en C.6, `second_order.py:53`) **non exporté**. Décision
  correcte : constante interne, valeur unique pour Blindsight/AGL student parity, les callers
  passent `weight_init_range=...` explicitement. Pas besoin d'API publique.
- ⚠️ **`distillation_loss` public mais dead code** (DETTE-3). La fonction est exposée via `__all__`
  mais `grep "distillation_loss(" src/` retourne 0 caller prod. Seul caller : `tests/unit/
  components/test_losses.py:14` (test de la fonction morte elle-même).

  **Décision** : garder public, cohérent avec DETTE-3 policy (preserve-paper-reproduction). Une
  démotion vers `_distillation_loss` private trahirait la parity student intentionnelle.

## `maps.networks/__init__.py` — backbone exports

2 symbols exportés :

```python
__all__ = ["FirstOrderMLP", "make_chunked_sigmoid"]
```

- ✅ Minimaliste, pas de fuite privée.
- ✅ Tous les callers src/ + tests utilisent `from maps.networks import ...`.

## Cross-check — imports callers vs `__all__`

Grep exhaustif `from maps.components|from maps.networks` dans `src/` et `tests/` :

| Import path                                                     | Caller                                          | Status |
|:----------------------------------------------------------------|:------------------------------------------------|:------:|
| `from maps.components import ComparatorMatrix, SecondOrderNetwork, WageringHead` | `tests/unit/components/test_second_order.py:8` | ✅ |
| `from maps.components import cae_loss, distillation_loss, wagering_bce_loss, weight_regularization` | `tests/unit/components/test_losses.py:12-17` | ✅ |
| `from maps.components import SecondOrderNetwork, cae_loss, wagering_bce_loss` | `blindsight/trainer.py:53`, `agl/trainer.py:51` | ✅ |
| `from maps.components import SecondOrderNetwork`                | `tests/parity/test_blindsight_pretrain.py:23`, `test_blindsight_reference.py:18`, `test_agl_pretrain.py:22` | ✅ |
| `from maps.networks import FirstOrderMLP`                       | `blindsight/trainer.py:59`, `agl/trainer.py:60`, parity tests × 4 | ✅ |
| `from maps.networks import FirstOrderMLP, make_chunked_sigmoid` | `tests/parity/test_agl_reference.py:8`          | ✅ |

**Direct module imports** (bypass `__init__.py`) :

| Import path                                                     | Caller                                          | Why bypass |
|:----------------------------------------------------------------|:------------------------------------------------|:----------:|
| `from maps.components.cascade import cascade_update, n_iterations_from_alpha` | `tests/unit/components/test_cascade.py:8` | test direct, OK |
| `from maps.components.cascade import cascade_update`            | `components/second_order.py:49`, `networks/first_order_mlp.py:32`, `sarl/model.py:34`, `sarl_cl/model.py:44` | **intra-package** — avoids circular import through package `__init__.py` |
| `from maps.components.losses import weight_regularization`      | `sarl_cl/trainer.py:62`                         | inconsistent style (see C16-F1) |
| `from maps.networks.first_order_mlp import make_chunked_sigmoid` | `agl/trainer.py:61`                            | inconsistent style (see C16-F1) |

✅ **Tous les imports __all__ sont correctement câblés.** Aucun nom fantôme, aucun caller cassé.

## Finding C16-F1 — Style inconsistency sur imports (mineur)

Deux callers utilisent le direct module path au lieu de l'API publique :

```python
# agl/trainer.py:60-61 (inconsistent)
from maps.networks import FirstOrderMLP                                  # public API ✅
from maps.networks.first_order_mlp import make_chunked_sigmoid           # direct module ⚠️

# sarl_cl/trainer.py:62 (inconsistent)
from maps.components.losses import weight_regularization                 # direct module ⚠️
# (could use: `from maps.components import weight_regularization`)
```

**Impact** : minime. Les 2 formes fonctionnent, mais l'inconsistance rend le refacto futur plus
risqué (si on déplace un symbol entre modules, les direct-path imports cassent).

**Piste C16-fix-1** : harmoniser les 2 imports vers `from maps.networks import ...` et
`from maps.components import ...` pour cohérence. 2 lignes à modifier, 0 breaking.

## Finding C16-F2 — `cascade_update` direct import intra-package

4 modules intra-package importent directement `from maps.components.cascade import cascade_update` :
- `components/second_order.py:49`
- `networks/first_order_mlp.py:32`
- `experiments/sarl/model.py:34`
- `experiments/sarl_cl/model.py:44`

**Pourquoi direct** : ces modules sont **importés PAR** `components/__init__.py` (transitively).
Si `second_order.py` faisait `from maps.components import cascade_update`, cela déclencherait
l'import de `maps.components.__init__.py` qui importe `second_order.py` lui-même → **circular
import**.

✅ **Pattern correct pour intra-package dependencies.** Pas un fix — piège connu Python.

## Finding C16-F3 — `__version__` obsolète

`src/maps/__init__.py:7` : `__version__ = "0.1.0"`. Jamais bumped depuis Sprint-00.

**Debate** :
- Pro bump : post Sprint-08 avec architecture bien plus mature (15+ commits C.1-C.15), → `0.2.0`.
- Contra : version semantic versioning peu significatif pour un fork de repro paper sans release
  externe. Pas de PyPI publish, pas de downstream consumer.

**Recommandation** : **skip C.16**. Bumper à `0.2.0` au moment de Phase G (final report) pour
marquer l'état "reproduction validated".

## Finding C16-F4 — Sprint-02 doc drift (historique, non-actionable)

`docs/sprints/sprint-02-extract-maps-core.md:83-84` mentionne :
```python
from maps.components import CascadeModule, SecondOrderNetwork
from maps.components.losses import contrastive_loss, wagering_bce_loss, distillation_loss
```

Noms obsolètes :
- `CascadeModule` → n'existe pas (on a `cascade_update` fonction, pas classe).
- `contrastive_loss` → renommé `cae_loss` en C.7 (D-002 naming collision resolved).

**Action** : aucune (doc historique Sprint-02, archivé). Si Rémy veut cleanup de la trail, Phase H.

## Fixes identifiées C.16

| ID        | Fix                                                                  | Scope                                       | Effort |
|:----------|----------------------------------------------------------------------|---------------------------------------------|:------:|
| C16-fix-1 | Harmoniser `agl/trainer.py:61` + `sarl_cl/trainer.py:62` imports vers API publique | 2 fichiers, 2 lignes         | 5 min  |
| C16-fix-2 | (skip) `__version__` bump                                            | —                                           | skip   |
| C16-fix-3 | (skip) sprint-02 doc cleanup                                         | —                                           | skip   |

## Cross-reference deviations.md

- **Aucune nouvelle deviation**.
- **DETTE-3** confirmée (`distillation_loss` public mais dead) — décision "keep public" validée.

## Résumé — Public API exports

- ✅ **Tous les `__all__` cohérents** avec les symbols importés (9 components + 2 networks).
- ✅ **Tous les callers** `from maps.components|networks import ...` pointent sur symbols
  exportés.
- ✅ **Pattern intra-package** (cascade_update direct import) correct pour éviter circular
  import.
- ⚠️ **1 inconsistance style** : 2 callers utilisent direct module paths — C16-fix-1 mineur.
- ⚠️ **`DEFAULT_WAGER_INIT_RANGE` non exposé** : correct, constante interne.
- ⚠️ **`distillation_loss` public mais dead code** : confirme DETTE-3 policy.
- ⚠️ **`__version__ = "0.1.0"`** : obsolète, bump à faire en Phase G.
- **0 bug bloquant, 1 fix style mineur, 2 skips debate.**

**C.16 clôturée. 1 fix pour batch C.17 ou direct.**
