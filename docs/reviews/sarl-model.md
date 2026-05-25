# Component review — `src/maps/experiments/sarl/model.py`

**Review sub-phase :** Sprint-08 D.3.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl/model.py` (147 L, 2 classes + 1 helper + 1 constant).
**Paper sources :** §3 (SARL), Table 11 (SARL hyperparams). Architecture suit Mnih et al. 2015 DQN
avec 1/4 filter count (MinAtar convention).
**Student sources :**
- `external/paper_reference/sarl_maps.py:129-182` (`size_linear_unit`, `num_linear_units`, `QNetwork`).
- `external/paper_reference/sarl_maps.py:245-` (`SecondOrderNetwork`) — reviewed C.4/C.5/C.6.
**Callers du port :**
- `sarl/training_loop.py:170-180` (instantiation policy + target).
- `sarl/trainer.py` (forward calls).
- `sarl_cl/model.py:44` (`SarlCLSecondOrderNetwork` hérite partiellement, cross-ref).

**DoD D.3** : doc créé, audit complet `SarlQNetwork` + récap `SarlSecondOrderNetwork`
(déjà reviewé), 0 code touché. Ce sub-phase est review-only, les fixes sont queued D.4 (cascade
no-op) et DETTE-1 (doublon déjà tracé).

---

## `_size_linear_unit` + `NUM_LINEAR_UNITS` (D.3-a)

Port :
```python
def _size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1

NUM_LINEAR_UNITS = _size_linear_unit(10) * _size_linear_unit(10) * 16
```

Student `sarl_maps.py:129-132` :
```python
def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1

num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
```

✅ **Bit-exact.**
- `size_linear_unit(10) = (10 - 2 - 1)//1 + 1 = 8`.
- `NUM_LINEAR_UNITS = 8 * 8 * 16 = 1024`. Conforme docstring sarl/data.py et shape
  `conv_out.view(B, -1) = (B, 1024)` dans forward.

**⚠️ `10` hardcodé** — MinAtar grid size. Les 5 games (space_invaders, breakout, seaquest, asterix,
freeway) utilisent toutes 10×10. Si une extension MinAtar ou un autre env utilise une taille
différente, le `NUM_LINEAR_UNITS` module-level serait faux. **Pas un bug actuel**, juste une
rigidité.

**Piste D3-fix-1** (info, skip D.3) : exposer `GRID_SIZE = 10` comme constante module ou calc
dynamique à partir de `env.state().shape`. Hors scope repro.

## `SarlQNetwork` (D.3-b) — conv + MLP + tied-weight decoder

Port L47-94 vs student L135-182 :

| Aspect                         | Student                                                        | Port                                                        | Match |
|:-------------------------------|:---------------------------------------------------------------|:------------------------------------------------------------|:-----:|
| `__init__(in_channels, num_actions)` | `Conv2d(in_ch, 16, kernel=3, stride=1)` + `fc_hidden(1024, 128)` + `actions(128, num_actions)` | idem | ✅ |
| Unused `self.sigmoid = nn.Sigmoid()` (student L151) | présent mais non-utilisé                                         | **absent** (propreté port)                                  | ⚠️ cosmetic — doesn't consume RNG (no weights), safe |
| Weight init                    | PyTorch default (`Kaiming uniform`)                            | idem (pas de `init.uniform_` explicite)                      | ✅    |
| Bias                           | True (default `nn.Linear`/`Conv2d`)                            | idem                                                         | ✅    |
| Forward shape pipeline         | `x (B, C, 10, 10) → ReLU+conv → (B, 16, 8, 8) → view (B, 1024) → ReLU+fc → (B, 128) → cascade → actions → (B, num_actions)` | idem | ✅ |
| Cascade on hidden              | `if prev_h2 is not None: Hidden = α*Hidden + (1-α)*prev_h2` inline | `cascade_update(hidden, prev_h2, cascade_rate)` (uses component primitive) | ✅ same math |
| Tied-weight decoder            | `Output_comparison = f.relu(f.linear(Hidden, fc_hidden.weight.t()))` | idem `F.relu(F.linear(hidden, self.fc_hidden.weight.t()))` | ✅ |
| Comparison                     | `Input - Output_comparison` (element-wise)                     | `flat_input - reconstruction`                                | ✅    |
| Return tuple                   | `(x, Hidden, Comparisson, Hidden)` (Hidden **twice**)          | `(q_values, hidden, comparison, hidden)` (hidden twice)      | ✅ student parity |

✅ **Port = student bit-exact** sur l'architecture + forward. L'absence du `self.sigmoid` unused
dans le port est une **divergence cosmétique safe** (pas de RNG consumption car `nn.Sigmoid()`
n'a pas de paramètres).

### (b1) Return tuple quadruple avec hidden×2 — pourquoi ?

Student L182 retourne `return x, Hidden, Comparisson, Hidden` — **`Hidden` dupliqué**. Les callers
(student `train()` L756+, notre port trainer L150+) font probablement
`q_values, h1, comparison, h2 = policy_net(...)` en décomposant la duplication comme deux
variables — mais ces **deux variables pointent sur le même tensor**.

**Pourquoi garder ?** Compat parity. Si on change en 3-tuple, les callers paper-parity cassent,
et les tests tier-1 forward doivent être mis à jour.

**Piste D3-fix-2** (debate, skip D.3) : documenter dans `deviations.md` comme DETTE-5 candidate.
Ou simplement garder — impact 0, confusion marginale. Je recommande **skip** — docstring
port suffit.

### (b2) 🚨 Cascade no-op sur `SarlQNetwork.forward` (C.1 finding confirmed)

Le forward `SarlQNetwork` :
```python
hidden = F.relu(self.fc_hidden(flat_input))  # deterministic — no dropout
hidden = cascade_update(hidden, prev_h2, cascade_rate)
```

**Pas de dropout dans le 1st-order SARL forward** (contrairement aux 2nd-order Blindsight/AGL qui
dropout C dans `SecondOrderNetwork`). Donc `hidden` est **déterministe** — à chaque appel avec le
même `x` + mêmes poids, la même valeur sort.

**Caller training_loop.py:289+** boucle `cascade_iterations_1` fois sur `policy_net(state, prev_h2,
cascade_rate)`. Au bootstrap `prev_h2=None` → `cascade_update` return `hidden`. Puis au step 2,
`prev_h2 = hidden_step1 = F(x, weights)`. Step 2 calcule `new_hidden = F(x, weights) = hidden_step1`
(déterministe), et `cascade_update(new, prev, α) = α·new + (1-α)·prev = α·hidden + (1-α)·hidden
= hidden`.

→ **Itérations 2 à N sont toutes égales à itération 1**. Cascade est mathématiquement **no-op**.

**Vérification rigoureuse** : documenté dans `docs/reviews/cascade.md §(d)` avec test analytique
`tests/unit/components/test_cascade.py::test_cascade_deterministic_caller_loop_is_1_iter_equivalent`
(ajouté C.2).

**Impact paper reproduction** : le paper annonce `cascade_iterations_1=50` (setting 2/4/6) comme
si ça avait un effet, mais sur le path SARL 1st-order c'est équivalent à 1 itération → compute
gaspillé 50× sans changement de résultat.

**Deux résolutions possibles** :
- **Option A** (paper-faithful) : garder `cascade_iterations_1=50` comme paramètre même si no-op —
  respect du paper même si compute gâché. Cohérent policy "paper = source of truth".
- **Option B** (compute-optimal) : shortcut `cascade_iterations_1=1` quand pas de dropout +
  documenté. Gain ~50× sur la 1st-order loop compute. **Risque** : si un futur path stochastique
  est ajouté (ex: noisy_net layers), le shortcut fausse.

**Piste D.4 fix (queued plan L509)** : *"Fix dropout-in-cascade si décision parity-safe"* — exactly
this trade-off. **Action D.3** : noter le paradoxe, pointer vers D.4 pour résolution.

### (b3) Tied-weight decoder — DETTE-2 cross-ref

Le tied-weight `F.linear(hidden, fc_hidden.weight.t())` rend `SarlQNetwork` spécial : pas de
decoder params séparés, tout est dans `fc_hidden`. C'est ce qui interdit l'utilisation de
`components.ComparatorMatrix` (C.3 finding) et justifie DETTE-2 (doublon cae_loss).

✅ **Architecture paper-faithful, divergence vs Blindsight/AGL intentionnelle.**

### (b4) `F.relu(F.linear(hidden, fc_hidden.weight.t()))` — ReLU decoder

Student applique `f.relu(f.linear(...))` — donc le decoder output est ReLU-activé. Port idem.
**Conséquence** : `reconstruction ≥ 0` toujours. `comparison = flat_input - reconstruction`
peut être négatif (si flat_input < reconstruction component-wise). OK pour le 2nd-order en aval
(pas de contrainte signe).

## `SarlSecondOrderNetwork` (D.3-c) — récap (déjà reviewé C.4/C.5/C.6)

Port L97-147, 51 L. Déjà audité :
- **C.4** : review (dropout 0.1, raw logits 2-unit, wager layer `NUM_LINEAR_UNITS → 2`).
- **C.6** : fixes appliqués (weight init uniform (0, 0.1), docstring, DETTE-1 flagged).
- **DETTE-1** : doublon `components.SecondOrderNetwork` vs `SarlSecondOrderNetwork` — justifié par
  tied-weight + raw logits + dims hardcodés. Unification Phase H.

**Rien à ajouter dans ce review D.3.**

## Init RNG order

Docstring port L71-72 : *"Layer construction order matches the paper for init RNG reproducibility
(PyTorch draws weights sequentially from the default generator)."*

Student order (L145-157) : `conv → sigmoid → fc_hidden → actions`.
Port order (L73-75)       : `conv → fc_hidden → actions`.

Le `nn.Sigmoid()` student est sans-paramètre donc **ne consomme pas de RNG**. Les ordres port et
student donnent donc **la même séquence de init RNG draws** : conv.weight → conv.bias →
fc_hidden.weight → fc_hidden.bias → actions.weight → actions.bias.

✅ **Init RNG parity preservée.**

## Cross-check Callers

| Caller                                | Instantiation args                             | Status |
|:--------------------------------------|:-----------------------------------------------|:------:|
| `sarl/training_loop.py:178-180`       | `SarlQNetwork(in_channels, num_actions)`       | ✅     |
| `sarl/trainer.py:...` (forward)       | `policy_net(state, prev_h2, cascade_rate)`     | ✅     |
| `sarl_cl/model.py` (héritage potentiel) | à reviewer D.17 (scope SARL+CL)               | deferred |

## Fixes identifiées D.3

| ID       | Fix                                                                | Scope                                  | Effort |
|:---------|--------------------------------------------------------------------|----------------------------------------|:------:|
| D3-fix-1 | (skip) exposer `GRID_SIZE = 10` comme constante ou dynamic         | `sarl/model.py:44`                     | skip   |
| D3-fix-2 | (skip) documenter quadruple return `(q, h, c, h)` — pas un fix mais un wart | —                           | skip   |
| D3→D.4   | Résolution cascade no-op : Option A (paper-faithful keep) vs B (shortcut) | plan D.4               | queued D.4 |

## Cross-reference deviations.md / reviews

- **C.1 cascade no-op sur forward deterministe** : **re-confirmée** côté SarlQNetwork. SARL 1st-order
  path = pas de dropout = cascade pure no-op. Voir `docs/reviews/cascade.md §(d)`.
- **DETTE-1** (SarlSecondOrderNetwork doublon) : confirmée architecture-wise — tied-weight
  decoder rend l'unification impossible sans refacto majeur.
- **DETTE-2** (cae_loss doublon) : le decoder ReLU + tied-weight justifie la version Huber
  hardcodée dans `sarl/losses.cae_loss` — architecturalement lié à `SarlQNetwork`.
- **Aucune nouvelle deviation** surfacée.

## Résumé — `sarl/model.py`

- ✅ **`NUM_LINEAR_UNITS=1024`** math bit-exact (8×8×16), hardcode `10` flaggé info-only.
- ✅ **`SarlQNetwork` architecture + forward + tied-weight decoder** bit-exact avec student
  (sauf `nn.Sigmoid()` unused supprimé, 0 impact RNG).
- ✅ **Init RNG parity** garantie malgré le Sigmoid student absent du port.
- 🚨 **Cascade no-op sur SARL 1st-order path** (C.1 re-confirmed) — résolution D.4.
- ⚠️ **Return `(q, h, c, h)` quadruple avec hidden×2** — student parity wart, docstring explicite.
- ✅ **`SarlSecondOrderNetwork` déjà reviewé** C.4/C.5/C.6 — rien à ajouter.
- **0 divergence paper-vs-port hors cascade paradox tracé ailleurs.**

**D.3 clôturée. 0 fix direct, 1 résolution queued D.4 (cascade no-op), 2 skips hors scope.**
