# Component review — `src/maps/components/cascade.py`

**Review sub-phase :** Sprint-08 C.1.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-19.
**File under review :** `src/maps/components/cascade.py` (79 lignes, 2 fonctions).
**Paper sources :** eq. 6 p. 6 (Know-Thyself cascade), eq. 9 p. 7 (SARL cascade).
**Student sources inline :**
- `external/paper_reference/sarl_maps.py:171, 238, 273` (Hidden, Output, comparison_out).
- `external/paper_reference/blindsight_tmlr.py:185, 247`.
- `external/paper_reference/agl_tmlr.py:536-567` (idem).
**Callers du port (grep) :**
- `src/maps/networks/first_order_mlp.py:110` (Blindsight/AGL path).
- `src/maps/components/second_order.py:148`.
- `src/maps/experiments/sarl/model.py:86, 145` (QNetwork + SecondOrder).
- `src/maps/experiments/sarl_cl/model.py:112, 174, 254`.

**DoD :** doc exists, 4 sections (a)-(d), fixes identified. Aucun code touché en C.1.

---

## (a) Conformité eq. 6 paper

**Paper eq. 6** (verbatim) :

$$a_{ir}(t) = \alpha \sum_j w_{ij} \, a_{js}(t) + (1 - \alpha) \, a_{ir}(t - 1)$$

Where : `α ∈ [0, 1]` cascade rate, `a_{ij}(t)` activation of neuron `i` in layer `j` at step `t`,
`w_{ij}` weight from `j` to `i`.

**Port** (`cascade.py:66`) :

```python
return alpha * new_activation + (1.0 - alpha) * prev_activation
```

**Mapping vérifié** :
- `α` → `alpha` (param).
- `Σ_j w_{ij}·a_{js}(t)` → `new_activation` (le caller a déjà calculé `ReLU(W·x + b)`).
- `a_{ir}(t-1)` → `prev_activation`.

**Bootstrap** (`cascade.py:64-65`) : `if prev_activation is None: return new_activation`. Cohérent
avec paper eq. 9 (SARL) qui dit *"`H^{(i)} = H^{(i)}_raw` if `H^{(i-1)} = None` otherwise α·new + (1-α)·prev"*.

✅ **Conformité sémantique à eq. 6 : exacte.**

### Note subtile — paper eq. 6 vs eq. 9

Paper eq. 6 (§2.1 Know-Thyself) est une **équation de dynamique temporelle continue** dérivée de
McClelland 1989 (cascade model). L'eq. 9 (§2.2 SARL) est une **discrétisation en boucle `for i =
0..N-1`** de cette dynamique. Le port implémente la discrétisation (eq. 9), cohérente avec ce que
font les 3 monolithes student.

**Subtilité possible** (voir (d) ci-dessous) : McClelland 1989 décrit une dynamique où **chaque
couche** cascade en parallèle, et `Σ_j w·a_{js}(t)` dépend du temps `t` parce que les activations
de la couche upstream `s` cascadent aussi. Dans notre port (et le student), **seule la dernière
hidden layer cascade** ; les couches upstream (conv, fc1) sont recalculées à l'identique chaque
itération depuis `x`. C'est une simplification — notée ici pour traçabilité, pas une divergence
paper-vs-port (le student fait pareil).

---

## (b) α bounds

**Port** (`cascade.py:58-63`) :

```python
if not (0.0 < alpha <= 1.0):
    raise ValueError(
        f"cascade alpha must lie in (0, 1]; got {alpha}. "
        "alpha=0 freezes the state (degenerate); negative/larger-than-1 "
        "values are not physically meaningful."
    )
```

**Paper** : `α ∈ [0, 1]` selon la formulation texte, mais sémantiquement :
- `α = 0` → `new = 0·new + 1·prev = prev` → freeze (dégénéré, aucun signal nouveau).
- `α = 1` → `new = 1·new + 0·prev = new` → collapse feed-forward, no cascade.
- `α ∈ (0, 1)` → vraie cascade.
- `α > 1` → non-physique (amplification, pourrait diverger).
- `α < 0` → non-physique.

**Port bounds `(0, 1]`** = strict `α > 0` + `α ≤ 1`. Rejette `α=0` (correct — papier dit α=0.02
dans la pratique) et accepte `α=1` (comme fallback feed-forward pour settings sans cascade).

**`n_iterations_from_alpha`** (`cascade.py:69-78`) : `int(1.0 / alpha)`. Pour `α=0.02` → 50.
Convention paper ("empirically selected 50 cascade iterations for all test cases (α = 0.02)",
§2.1 p. 6). ✅

✅ **α bounds cohérents avec sémantique paper et usage pratique.**

---

## (c) Cohérence API cross-domaine

### Signature uniforme

Tous les callers utilisent la même signature `cascade_update(new, prev, alpha)` :

| Caller                                          | Contexte                                    | `new`         | `prev`           |
|-------------------------------------------------|---------------------------------------------|--------------|-------------------|
| `sarl/model.py:86`                              | QNetwork forward (1st order)                | `hidden` (128-d) | `prev_h2`     |
| `sarl/model.py:145`                             | SecondOrderNetwork forward                  | `dropout(comparison_matrix)` | `prev_comparison` |
| `sarl_cl/model.py:112, 254`                     | QNetwork + AdaptiveQNetwork CL              | `output` (128-d) | `prev_h2`     |
| `sarl_cl/model.py:174`                          | SecondOrderNetwork CL                       | `comparison`  | `prev_comparison` |
| `components/second_order.py:148`                | shared SecondOrderNetwork (Blindsight/AGL)  | `comparison`  | `prev_comparison` |
| `networks/first_order_mlp.py:110`               | shared FirstOrderMLP (Blindsight/AGL)       | `h2` (décoder output) | `prev_h2` |

✅ **Signature cohérente cross-domaine.** Tous les callers passent la fresh activation post-
activation-function ; `prev` propagé depuis l'itération précédente (ou `None` au bootstrap).

### Factorization vs student inline

Les 3 monolithes student font **la même mathématique inline**, avec du code dupliqué à chaque site
(6+ endroits dans sarl_maps.py, ~4 dans blindsight_tmlr.py, ~4 dans agl_tmlr.py) :

```python
# sarl_maps.py:171
Hidden = cascade_rate * Hidden + (1 - cascade_rate) * prev_h2
# sarl_maps.py:238
Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2
# blindsight_tmlr.py:185
h2 = cascade_rate * h2 + (1 - cascade_rate) * prev_h2
```

Notre port **factorise** cette logique dans une helper unique (`cascade.py:30`) + utilise depuis 6+
callsites. ✅ **Amélioration maintenabilité nette, pas de divergence sémantique.**

### Confusion potentielle avec α EMA wagering (⚠️)

`α` apparaît **deux fois** dans le papier avec des valeurs différentes :
- **α cascade** = 0.02 (eq. 6, paper §2.1 p. 6).
- **α EMA wagering** = 0.45 (Table 11 p. 30, eq. 13 p. 8).

Notre port utilise `cascade_rate` comme nom de paramètre à tous les callsites — pas de confusion
côté code. Mais le nom de la fonction `cascade_update(..., alpha)` peut prêter à confusion lors de
lecture rapide. **Piste d'amélioration mineure : renommer `alpha` en `cascade_rate` dans la
signature `cascade_update(new, prev, cascade_rate)`** pour cohérence avec les callers. Non-bloquant.

⚠️ **Amélioration mineure identifiée : uniformiser naming `cascade_rate` dans la signature.**

---

## (d) Pistes optim — et 🚨 finding semantique majeur

### Finding 🚨 — cascade ≡ no-op pour 1st-order SARL

**Analyse mathématique** :

SARL QNetwork forward (`sarl/model.py:77-94`) :
```python
def forward(self, x, prev_h2, cascade_rate):
    conv_out = F.relu(self.conv(x))
    flat_input = conv_out.view(...)
    hidden = F.relu(self.fc_hidden(flat_input))  # ← H_raw, déterministe depuis x
    hidden = cascade_update(hidden, prev_h2, cascade_rate)
    q_values = self.actions(hidden)
    ...
```

**Pas de dropout dans SarlQNetwork** (grep confirmé). `H_raw = ReLU(W·x + b)` est donc
**déterministe** et **identique à chaque itération** de la boucle caller.

Caller loop (`sarl/trainer.py:160`) :
```python
for _ in range(cascade_iterations_1):  # 50 pour setting 2 ou 4 ou 6
    q_policy, h1, comparison_1, main_task_out = policy_net(states, main_task_out, cascade_rate_1)
```

Trace des valeurs de `hidden` produites :
- **i=0** (`prev=None`) : `hidden = H_raw` (bootstrap).
- **i=1** (`prev=H_raw`) : `hidden = α·H_raw + (1-α)·H_raw = H_raw`.
- **i=2** : `hidden = α·H_raw + (1-α)·H_raw = H_raw`.
- **i=N-1** : `hidden = H_raw`.

**La cascade sur un réseau déterministe est un no-op mathématique : toutes les itérations
produisent la même valeur qu'une itération unique.**

**Implication** : SARL Setting 2 (cascade on 1st Net only, `cascade_iter_1=50, cascade_iter_2=1,
meta=False`) devrait produire des chiffres **mathématiquement identiques** à Setting 1 (baseline
no cascade). Paper Table 6 montre pourtant des chiffres significativement différents (Seaquest
Setting 2 Val Z = **−2.59** vs Setting 1). **Paradoxe à résoudre.**

### Hypothèses pour le paradoxe

1. **H1 — Stochasticité externe** : la boucle `for i in range(50)` consomme potentiellement des
   appels RNG (sélection d'indices, shuffle, etc.) même si le forward est déterministe. Une
   désynchronisation des seeds entre Setting 1 et Setting 2 pourrait expliquer des chiffres
   différents, dominés par la variance. Compatible avec Seaquest Setting 1 mean=1.21 ± 0.16 et
   Setting 2 mean=0.76 ± 0.19 : des `z=-2.6` avec N=3 seeds est très sensible au tirage.
2. **H2 — cascade sur un autre path** : peut-être la cascade modifie autre chose qu'on a pas vu
   (buffer, gradient, target network sync timing, scheduler step count). À tracer Phase D.
3. **H3 — paper lui-même bruité** : avec N=3 seeds, Setting 1 vs Setting 2 paper peut être du
   bruit statistique (confidence interval large).

### Finding ⚠️ — cascade est **active** pour 2nd-order (dropout)

SARL SecondOrderNetwork forward (`sarl/model.py:138-147`) :
```python
comparison_out = self.dropout(comparison_matrix)  # ← stochastique
comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
```

Ici `comparison_out` **varie** à chaque itération (dropout resample le mask). La cascade moyenne
effectivement **50 réalisations stochastiques** → réduction de variance équivalente à un
"dropout moyenné", différent de `train()` mode classique.

Idem **Blindsight** (`blindsight_tmlr.py:184-187`) : `encoder` a un `dropout(0.1)`. Chaque
itération resample. Cascade = vraie moyenne.

**Donc** : cascade est **active** pour les chemins **stochastiques** (dropout, BN, etc.) mais
**no-op** pour les chemins **déterministes**. **Dans notre port** :

| Chemin                                  | Dropout dans forward ? | Cascade is... |
|-----------------------------------------|:---------------------:|:-------------:|
| SARL QNetwork                           | Non                    | ⚠️ NO-OP      |
| SARL SecondOrderNetwork                 | Oui (p=0.1)            | ✅ active      |
| SARL+CL QNetwork + AdaptiveQNetwork     | Non (à vérifier D.17)   | ⚠️ NO-OP      |
| SARL+CL SecondOrderNetwork              | Oui (probable)          | ✅ active      |
| Blindsight FirstOrderMLP                | Oui (encoder)           | ✅ active      |
| Blindsight SecondOrderNetwork           | Oui                     | ✅ active      |
| AGL FirstOrderMLP                       | Oui (à vérifier D.27)   | ✅ active      |
| AGL SecondOrderNetwork                  | Oui                     | ✅ active      |

### Pistes d'optim

#### Opt-C1-A : skip cascade inner loop pour 1st-order SARL (and deterministic paths)

- **Statut** : safe iff SARL QNetwork reste déterministe (no dropout added future).
- **Effort** : trivial — remplacer `for _ in range(cascade_iterations_1): policy_net(...)` par un
  seul appel `policy_net(..., prev=None, ...)`.
- **Impact estimé** : **énorme** pour Setting 2 (50× speedup forward QNetwork). Moindre pour
  Setting 4/6 (la cascade 2nd-order est utile → peut pas skip le tout).
- **Risque** : **parité paper** — si paper Table 6 Setting 2 ≠ Setting 1 est **intentionnel**
  (pas un artefact de bruit RNG), on n'a pas le droit de skip car on perd ce "côté bénéfice
  cascade 1st" revendiqué. **Mais mathématiquement**, nos observations disent que les deux DOIVENT
  être identiques modulo RNG path.
- **Action proposée Phase D** : **garder la loop** (policy paper-faithful), mais **ajouter un test
  analytique** qui vérifie que les 50 forwards produisent bien la même valeur quand
  dropout=off → sanity test de cette review.

#### Opt-C1-B : analytical closed-form pour cascade déterministe

Si on confirme no-op pour path déterministe :
$$H^{(n)} = \alpha \cdot H_{raw} \cdot \sum_{k=0}^{n-1} (1-\alpha)^k + (1-\alpha)^n \cdot H^{(0)}
= H_{raw} \cdot (1 - (1-\alpha)^n) + (1-\alpha)^n \cdot H^{(0)}$$

Avec bootstrap `H^{(0)} = H_raw`, ça se simplifie en `H^{(n)} = H_raw`. **Cascade = no-op en
fermé, confirmé**.

#### Opt-C1-C : vectoriser cascade pour path stochastique (dropout)

Le speedup potentiel est limité — la boucle fait 50 forward passes qui sont **par construction
séquentiels** (prev dépend de i-1). Pas de parallelism trivial.

Mais : on pourrait faire **50 dropout samples en parallèle** (shape (50, B, D) au lieu de
séquentiel), puis appliquer la récurrence cascade sur le buffer 50× en CPU fast (récurrence simple).
Gain théorique modeste (reduction de kernel launch overhead sur le dropout). **Pas prioritaire.**

#### Opt-C1-D : renommer `alpha` → `cascade_rate` dans la signature

Trivial. Améliore la clarté. À faire si on touche à `cascade.py` en D.

### Fixes identifiées (pour Phase D follow-up)

| ID              | Fix                                                                  | Scope               | Effort |
|-----------------|----------------------------------------------------------------------|--------------------|:------:|
| C1-fix-1        | Ajouter test analytique : cascade déterministe → identique à 1 iter  | `tests/unit/components/test_cascade.py` | 15 min |
| C1-fix-2        | Investiguer paradoxe Setting 1 vs 2 SARL (H1/H2/H3 ci-dessus)        | Phase D.7 SARL trainer review | 1-2h |
| C1-fix-3        | Renommer `alpha` → `cascade_rate` dans `cascade_update()` signature  | `cascade.py` + callers | 20 min |
| C1-fix-4 (⚠️)   | Décision policy : skip la loop cascade sur path déterministe ?         | Phase D.4 ou D.7    | Rémy |

### C.1 NE touche pas au code

Les 4 fixes ci-dessus sont identifiées, pas appliquées. Application en Phase D (review per-experiment).

---

## Résumé — `cascade.py`

- **Sémantique** : ✅ conforme à paper eq. 6 + 9 + McClelland 1989.
- **API cross-domaine** : ✅ factorisation propre (vs student inline dupliqué), signature
  uniforme sur 6+ callsites.
- **α bounds** : ✅ `(0, 1]` correct.
- **🚨 Finding critique** : la cascade est un **no-op mathématique** sur les chemins **déterministes**
  (SARL QNetwork, probablement SARL+CL QNetwork). Elle est **active** uniquement sur les chemins
  **stochastiques** (dropout). **Paradoxe** avec paper Table 6 SARL Setting 1 ≠ Setting 2 à
  résoudre Phase D.
- **4 fixes identifiées** non-appliquées (Phase D).

**C.1 clôturée.**
