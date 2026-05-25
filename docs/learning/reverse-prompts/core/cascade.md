# Reverse-prompt — `src/maps/core/cascade.py`

**Module path actuel (sur main) :** `src/maps/components/cascade.py` (renommé en
`core/cascade.py` au Sprint 11+).
**Taille :** 79 lignes, 2 fonctions publiques.
**Paper :** §2.1 eq.6 (cascade dynamics), §2.2 eq.9 (SARL discretization).
**Référence externe :** McClelland (1979, 1989) — cascade model.
**Review existant :** `docs/reviews/cascade.md` (Sprint-08 C.1, 290 lignes —
très bon point de départ, on capitalise dessus).

---

## 1. Quelle aurait été la spec ?

**Prompt qu'on aurait donné à Claude :**

> Écris un module Python `cascade.py` qui implémente le cascade model de
> McClelland (1989) tel que repris en équation 6 du paper MAPS. Le module
> doit fournir une fonction `cascade_update(new, prev, α)` qui applique
> une itération de la récurrence
>
>   a(t) = α · f(x) + (1 - α) · a(t-1)
>
> avec α ∈ (0, 1]. Quand `prev is None` (bootstrap à t=0), retourne
> `new` directement. Le module doit aussi fournir une helper
> `n_iterations_from_alpha(α) → int(1/α)` utilisée par les domaines
> Blindsight et AGL pour fixer 50 itérations à α=0.02.
>
> Cette fonction sera appelée depuis ~6 endroits différents :
> - `networks/first_order_mlp.py` (chemin Blindsight/AGL — encoder)
> - `core/second_order.py` (chemin comparator + dropout)
> - `domains/sarl/model.py` (Q-network + 2nd-order)
> - `domains/sarl_cl/model.py` (Q-network + adaptive Q + 2nd-order)
>
> La signature doit être uniforme cross-domain. Les callers passent
> toujours `(new_activation, prev_activation, rate)`.
>
> Contraintes :
> - Type hints exhaustifs (Python 3.12+, `from __future__ import
>   annotations`).
> - Docstring NumPy-style avec citation paper et McClelland.
> - Validation `α ∈ (0, 1]` (raise ValueError sinon).
> - Pas de side effects, pas d'allocation tensor inutile.
> - Pure-PyTorch (pas de numpy intermédiaire).

**Niveau de spec :** API + sémantique + références. Pas de détails
d'implémentation (l'implémentation se résume à 1 ligne).

## 2. Contraintes scientifiques

### Équation 6 du paper (verbatim)

$$a_{ir}(t) = \alpha \sum_j w_{ij} \, a_{js}(t) + (1 - \alpha) \, a_{ir}(t-1)$$

où :
- `α ∈ [0, 1]` : taux d'intégration cascade
- `a_{ij}(t)` : activation du neurone `i` dans la couche `j` au temps `t`
- `w_{ij}` : poids de `j` vers `i`

**Mapping au code :**

| Paper | Code |
|-------|------|
| `α` | `cascade_rate` (param) |
| `Σ_j w_{ij} · a_{js}(t)` | `new_activation` (calculée par le caller) |
| `a_{ir}(t-1)` | `prev_activation` |
| Résultat `a_{ir}(t)` | Valeur de retour |

### Hyperparamètres du paper

- **α = 0.02** (paper §2.1 p.6, "empirically selected")
- **50 itérations** = `int(1/α) = 50` (convention paper)
- Différencier soigneusement de l'autre α qui apparaît dans le paper :
  **α EMA wagering = 0.45** (Table 11 p.30, eq. 13 p.8). Ces deux α n'ont
  rien à voir. D'où le nom `cascade_rate` à l'usage.

### Note paper-vs-implémentation subtile

McClelland (1989) décrit une dynamique où **toutes les couches** d'un
réseau cascadent en parallèle. Dans l'implémentation MAPS (et dans le code
original Juan), **seule la dernière hidden layer cascade** ; les couches
upstream (conv, fc1) sont recalculées à l'identique chaque itération.
C'est une simplification — pas une divergence paper-vs-port (le student
fait pareil), mais à mentionner pour la traçabilité.

## 3. Contraintes d'ingénierie

### Factorisation (vs. duplication student)

Le code student de Juan duplique la maths inline 14+ fois sur les 3
monolithes :

```python
# external/paper_reference/sarl_maps.py:171
Hidden = cascade_rate * Hidden + (1 - cascade_rate) * prev_h2

# external/paper_reference/sarl_maps.py:238
Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2

# external/paper_reference/blindsight_tmlr.py:185
h2 = cascade_rate * h2 + (1 - cascade_rate) * prev_h2

# ... 11 autres duplications
```

Le port (et la nouvelle réécriture) factorise dans une seule fonction.
**C'est un gain de maintenabilité net, pas une divergence sémantique.**
Test parity : on doit produire bit-for-bit le même tensor que le student
inline (modulo l'ordre des opérations float qui est respecté par cette
implémentation).

### Bounds checking

`α=0` est mathématiquement légal mais physiquement dégénéré (freeze :
`new = 0·new + 1·prev = prev`, aucune information nouvelle ne s'intègre).
Le code rejette avec `ValueError`. À garder.

`α=1` est l'option "no cascade" (collapse feed-forward : `new = 1·new +
0·prev = new`). Le code l'accepte — utile comme fallback dans les settings
factoriels où cascade est OFF.

### API contract

Tous les callers utilisent la même signature positionnellement, donc
**renommer `alpha` → `cascade_rate` dans la signature** est safe (et
recommandé pour clarté). C'est la fix C1-fix-3 du review.

## 4. Deviations / dettes présentes

### 🚨 Finding semantique majeur — D-sarl-cascade-noop

**Origine :** review `docs/reviews/cascade.md §(d)` (Sprint-08 C.1) et
deviations.md (`D-sarl-cascade-noop`).

Sur les chemins **déterministes** (pas de dropout dans le forward), la
cascade est un **no-op mathématique** :

- t=0 (bootstrap) : `h = H_raw`
- t=1 : `h = α·H_raw + (1-α)·H_raw = H_raw`
- t=N : `h = H_raw`

Donc 50 itérations de cascade sur un forward déterministe (e.g.
`SarlQNetwork` qui n'a pas de dropout) **produisent exactement la même
valeur qu'une itération unique**. C'est une fermeture analytique
démontrable.

**Active vs no-op selon le chemin :**

| Chemin | Dropout dans forward ? | Cascade is... |
|--------|:---------------------:|:-------------:|
| SARL QNetwork (1st order) | Non | ⚠️ NO-OP |
| SARL SecondOrderNetwork | Oui (p=0.1) | ✅ active |
| SARL+CL QNetwork + AdaptiveQNetwork | Non | ⚠️ NO-OP |
| SARL+CL SecondOrderNetwork | Oui | ✅ active |
| Blindsight FirstOrderMLP | Oui (encoder p=0.1) | ✅ active |
| Blindsight SecondOrderNetwork | Oui (p=0.5) | ✅ active |
| AGL FirstOrderMLP | Oui | ✅ active |
| AGL SecondOrderNetwork | Oui | ✅ active |

**Implication paper :** SARL Setting 2 (cascade 1st-order only) devrait
produire des chiffres identiques à Setting 1 (baseline no cascade), à RNG
près. Paper Table 6 montre pourtant Seaquest Setting 2 z=−2.59 vs
Setting 1. Soit (H1) l'écart est du bruit RNG (N=3 seeds), soit (H2) la
cascade a un side-effect non-trivial qu'on n'a pas identifié, soit (H3)
le paper lui-même est bruité. Phase D Sprint-08 a tranché : **garder la
boucle, paper-faithful, logger un warning**.

### ⚠️ D-marl-cascade-not-implemented

Le paper Table 12 admet implicitement que MARL n'utilise pas la cascade
(cascade_iter=1 forcé dans le code étudiant). Pas une déviation au sens
strict, juste une limitation paper-admise.

### Pas de dette technique sur ce module

`cascade.py` est petit, bien typé, bien testé. Aucune DETTE-* ouverte.
Pas de hardcoding (α est paramétrable depuis `config/maps.yaml`).

## 5. Questions ouvertes

> *Cette section est délibérément ouverte. À enrichir par Rémy en
> lisant le code et la doc, ou via Claude chat.*

- **Q1 :** Pourquoi McClelland a-t-il choisi cette forme exacte de
  cascade plutôt qu'une moyenne mobile exponentielle "classique" ? Y
  a-t-il une justification neuroscientifique au facteur α devant
  `new_activation` plutôt que devant `prev` (i.e. pourquoi pas la
  convention inverse) ?

- **Q2 :** Le no-op finding sur SARL — est-ce que Juan/Antoine sont au
  courant ? La review C.1 conclut "garder paper-faithful + warning",
  mais à long terme, c'est une zone d'ambiguïté qui mériterait d'être
  clarifiée scientifiquement (cascade + déterministe = quoi exactement ?).

- **Q3 :** Pour l'optimisation perf — peut-on **prouver** que skipper la
  boucle quand le chemin est déterministe ne change RIEN aux outputs
  (modulo état RNG) ? Si oui, on peut accélérer Setting 2 SARL de ~50×.
  Test analytique à écrire (C1-fix-1) avant tout skip.

- **Q4 :** *(à toi, Rémy : qu'est-ce qui te paraît non-obvious en lisant
  ce module ?)*

## 6. Pour le rewrite (Sprint 11+)

### À garder tel quel
- Sémantique exacte de eq. 6 / eq. 9 — `α · new + (1-α) · prev`
- Bootstrap `if prev is None: return new`
- Bounds check `α ∈ (0, 1]`
- `n_iterations_from_alpha(α) = int(1/α)` helper

### Petites améliorations
- **Renommage paramètre `alpha` → `cascade_rate`** dans la signature
  (clarté + cohérence avec les callers). C1-fix-3 du review.
- **Type hints `Tensor | None`** (utiliser le pipe operator, requiert
  `from __future__ import annotations`).
- **Citations dans la docstring** : ajouter référence Vargas2025 et
  McClelland1989 en format `[Author1989]_`.

### Tests à écrire (avant de toucher au code)
- **Test analytique no-op** (C1-fix-1) : `cascade_update` appliquée N
  fois sur un tensor déterministe identique produit exactement le même
  tensor que 1 application. Vérifie la fermeture analytique.
- **Test parity bit-for-bit** vs `external/paper_reference/sarl_maps.py`
  inline mathématique. Reconstruire ce test depuis zéro (le test parity
  actuel sera supprimé).
- **Test bounds** : `α=0` raise, `α=-0.1` raise, `α=1.1` raise, `α=1.0`
  OK, `α=0.02` OK.

### Opportunités perf identifiées (à vérifier au moment du rewrite)
- **Skip boucle si chemin déterministe** : Si on confirme analytiquement
  que SARL Q-network est déterministe, on peut court-circuiter la boucle
  de 50 itérations en 1 appel. ~50× speedup local pour Setting 2.
  **À discuter** : voulons-nous faire ça (et trahir paper-faithful) ou
  garder la boucle pour parity stricte ? La review C1 dit "garder".
- **Sampler dropout 50 fois en parallèle** sur path stochastique (cascade
  + dropout) : préallouer un buffer `(50, B, D)` au lieu d'appels
  séquentiels. Gain modeste (overhead kernel launch), pas prioritaire.

### Connexion avec d'autres modules
- Appelé par `core/second_order.py` (1 site)
- Appelé par `networks/first_order_mlp.py` (1 site)
- Appelé par `domains/sarl/model.py` (2 sites — Q + 2nd-order)
- Appelé par `domains/sarl_cl/model.py` (3 sites — Q, AdaptiveQ, 2nd-order)
- Référencé conceptuellement par tous les trainers

## Méta — pour une session Claude chat

Si je veux explorer cette dynamique avec Claude chat (web), bon prompt
à coller :

```text
J'étudie le cascade model de McClelland (1989) tel que repris dans le
paper MAPS (Vargas et al. 2025, TMLR submission) à l'équation 6. La
dynamique est :

    a(t) = α · f(x) + (1 - α) · a(t-1)

avec α = 0.02 et 50 itérations.

J'ai observé que sur un forward déterministe (sans dropout), 50
itérations de cascade produisent strictement la même valeur qu'une seule
itération (fermeture analytique : a(∞) → f(x) géométriquement). Pourtant
le paper Table 6 SARL Setting 2 (cascade 1st-order only) montre des
chiffres différents de Setting 1 (no cascade).

Trois hypothèses :
1. Bruit RNG (N=3 seeds, sensible)
2. Side-effect non-identifié (buffer ? scheduler ?)
3. Paper bruité

Comment investiguer rigoureusement cette ambiguïté ? Quels tests
analytiques poser ? Y a-t-il dans la littérature des discussions du
cascade model sur réseaux déterministes vs stochastiques ?
```
