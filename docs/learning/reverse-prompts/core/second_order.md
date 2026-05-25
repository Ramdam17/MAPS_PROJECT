# Reverse-prompt — `src/maps/core/second_order.py`

**Module path actuel (sur main) :** `src/maps/components/second_order.py`
(renommé `core/second_order.py` en Sprint 11+).
**Taille :** 217 lignes, 3 classes (`ComparatorMatrix`, `WageringHead`,
`SecondOrderNetwork`).
**Paper :** §2.1 eq.1 (comparator), eq.2 (dropout), eq.3 (wager linear),
eq.5 (BCE wager loss). §2.2 cite Pasquali & Cleeremans (2010) et Koch &
Preuschoff (2007).
**Review existant :** `docs/reviews/second_order.md` (Sprint-08 C.3/C.4/C.5,
~500 lignes, très exhaustif).

---

## 1. Quelle aurait été la spec ?

**Prompt qu'on aurait donné à Claude :**

> Écris un module Python `second_order.py` qui implémente le second-order
> network du paper MAPS (réseau métacognitif observant le first-order et
> produisant un "wager" — confiance dans la décision first-order). Trois
> classes :
>
> **1. `ComparatorMatrix`** (eq.1) : `nn.Module` stateless qui calcule
> `C = first_order_input - first_order_output` (différence élément-wise).
> Bien que sans paramètres, garde une `nn.Module` pour symétrie API et
> pour permettre de la remplacer plus tard par un comparator appris.
> Valide que les 2 tensors ont la même shape (raise `ValueError` sinon —
> torch broadcasterait silencieusement sinon).
>
> **2. `WageringHead`** (eq.3) : linear readout `(input_dim, n_wager_units)`
> qui prend la sortie post-dropout post-cascade du comparator. Deux modes :
> - `n_wager_units=1` (default, reproduction) : sigmoid → probabilité [0,1]
> - `n_wager_units=2` (paper-faithful) : raw logits, downstream loss
>   applique sigmoid per-unit (`F.binary_cross_entropy_with_logits`)
>
> Param optionnel `hidden_dim` : si > 0, insère un `Linear → ReLU` avant
> le readout final (Pasquali & Cleeremans 2010 — paper §2.2 cite cette
> archi mais le code student passait `hidden_2nd` sans s'en servir, bug
> qui plafonnait la wager accuracy ; cf. D.25). Default 0 = chemin
> student code.
>
> Init uniforme `(0, 0.1)` sur les weights du readout, biases laissés au
> default PyTorch (matches student exactement).
>
> **3. `SecondOrderNetwork`** (chapeau) : compose les 3 étapes paper :
> `Comparator → Dropout → cascade_update → WageringHead`. Forward signature
> `(first_order_input, first_order_output, prev_comparison, cascade_rate)`
> et retourne `(wager, comparison_out_post_cascade)`. Le caller threade
> `comparison_out` en `prev_comparison` à l'itération suivante.
>
> Le `SecondOrderNetwork` est partagé Blindsight + AGL. SARL a sa propre
> classe (`SarlSecondOrderNetwork`) avec tied-weight decoder, dropout
> p=0.1 au lieu de 0.5, et dimensions hardcodées — duplication intentionnelle
> documentée comme DETTE-1.

## 2. Contraintes scientifiques

### Équations paper

**eq.1** (comparator) :
$$\mathbf{C}_t = \mathbf{X}_t - \hat{\mathbf{Y}}^{(1)}_t$$
où `X_t` = input au first-order, `Ŷ^{(1)}_t` = output du first-order
(reconstruction).

**eq.2** (dropout) :
$$\mathbf{C}'_t = \text{Dropout}(\mathbf{C}_t)$$

**eq.3** (wager linear) :
$$\mathbf{W}_t = \mathbf{W} \cdot \mathbf{C}'_t + \mathbf{b}$$
— **raw logits, pas d'activation dans l'équation**. L'activation est
appliquée par la loss eq.5 (sigmoid via BCE-with-logits).

**eq.5** (BCE wager loss) :
$$\mathcal{L}_{BCE} = -[y \cdot \log(\sigma(W)) + (1-y) \cdot \log(1 - \sigma(W))]$$

### Citations paper

- **Pasquali & Cleeremans (2010)** : "Know thyself: Metacognitive networks
  and measures of consciousness". Cognition. Origin de l'archi
  Comparator + Wager hidden layer. **Cite hidden layer en paper §2.2
  ("as in Pasquali & Cleeremans 2010")** mais student code l'oublie.
- **Koch & Preuschoff (2007)** : "Betting the house on consciousness".
  Origine du concept de 2-unit wager (bet vs no-bet binaire).

### Hyperparamètres

- **Dropout = 0.5** Blindsight/AGL (student `blindsight_tmlr.py:222`)
- **Dropout = 0.1** SARL (`sarl_maps.py:254`) — domaine séparé,
  `SarlSecondOrderNetwork`
- **Weight init = uniform(0, 0.1)** sur les 3 domaines (paper §2.1 silencieux,
  student cohérent)

## 3. Contraintes d'ingénierie

### Composition ordre = paper

```
forward(fi, fo, prev_comparison, α):
    comparison_matrix = self.comparator(fi, fo)       # eq.1
    comparison_out = self.dropout(comparison_matrix)  # eq.2
    comparison_out = cascade_update(comparison_out,   # eq.6 (cascade)
                                    prev_comparison, α)
    wager = self.wagering_head(comparison_out)        # eq.3
    return wager, comparison_out                       # ← post-cascade
```

L'ordre `dropout → cascade → wager` est paper-faithful via student.
Le **retour post-cascade** est essentiel — c'est ce que le caller doit
threader en `prev_comparison` à t+1 pour que la cascade converge.

### Cascade + Dropout = averaging

Le dropout *à l'intérieur* de la cascade fait que chaque itération
voit un **mask différent**. La cascade accumule donc :
$$\sum_{t=1}^{N} (1-α)^{N-t} · \text{Dropout}_t(C)$$
ce qui **moyenne** ~50 dropout masks → réduction de variance équivalente
à un Monte-Carlo dropout (Gal & Ghahramani 2016). C'est **le mécanisme
qui rend la cascade non-triviale** sur le 2nd-order Blindsight/AGL.

Sur SARL Q-network (pas de dropout), la cascade redevient un no-op
(voir `cascade.md` D-sarl-cascade-noop).

### Gradient à travers 50 cascade steps

Le caller fait `loss.backward(retain_graph=True)` à travers la boucle
de 50 itérations cascade. Conséquences :
- **Memory ×50** : le graph garde 50 copies intermédiaires
- **Backward depth** : 50 multiplications-chaînées + dropouts + linears
- **Régularisation implicite** : 50 dropout masks indépendants moyennés
  dans le gradient

Comportement paper-faithful (matches student qui a le même unroll).
**Optimisation potentielle** non explorée : détacher `prev_comparison`
après N étapes pour borner la profondeur du graph.

### Stateless `ComparatorMatrix` vs `nn.Module`

Choix design : `ComparatorMatrix` n'a aucun paramètre, donc pourrait
être une simple fonction. Mais elle est en `nn.Module` pour :
- Symétrie avec WageringHead et SecondOrderNetwork
- Extensibilité (on peut la remplacer par un comparator appris)
- Cohérence des side-effects (registry dans le `state_dict()`, etc.)

## 4. Deviations / dettes présentes

### D-001 — Wager units sigmoid vs raw logits

Paper §2.1 eq.3 + eq.5 : 2 units **raw logits**, sigmoid INSIDE la BCE
loss. Student code : 1 unit sigmoid sur tout (Blindsight/AGL) ou 2
units raw logits (SARL — sa propre classe).

Notre port : `n_wager_units=1` par défaut (parity student Blindsight/AGL),
`n_wager_units=2` disponible mais **rarement plumb dans les callers**.

Notes :
- 1-unit sigmoid ≡ 2-unit raw logits avec targets one-hot (mathématiquement
  équivalents pour binaire 1-hot)
- Donc pas un cause de gap RG-002 (cf. D.25 H8 ablation)

### D-blindsight-wager-hidden (D.25) + D-agl-wager-hidden

Paper §2.2 dit "as in Pasquali & Cleeremans (2010)" → implique hidden
layer dans le wager. Student code prend un paramètre `hidden_2nd` mais
**ne l'utilise jamais** (bug L222 `blindsight_tmlr.py`, ne le passe pas
au `nn.Linear`).

Sans hidden layer : wager plafonne à 0.67 (Blindsight). Avec hidden
layer (`hidden_dim=100` pour Blindsight, `hidden_dim=48` pour AGL) :
wager monte à 0.82 (paper 0.85).

**Résolution Sprint-08 D.25** : `hidden_dim` paramétrable, default 0
(student behavior), Blindsight active à 100, AGL à 48. **Pasquali hidden
restored.**

### DETTE-1 — Doublon `SecondOrderNetwork` vs `SarlSecondOrderNetwork`

Deux classes avec 80% de chevauchement :

| Aspect | `core.SecondOrderNetwork` (BS/AGL) | `sarl.SarlSecondOrderNetwork` |
|--------|-----------------------------------|-------------------------------|
| ComparatorMatrix | Oui (instance) | Non (inline, tied-weight) |
| Dropout | 0.5 | 0.1 |
| Wager dims | `(input_dim, n_wager_units)` | `(1024, 2)` hardcoded |
| Wager activation | sigmoid (n=1) | raw logits |
| Forward sig | `(fi, fo, prev, α)` | `(comparison, prev, α)` |

**Pourquoi pas unifier** : SARL utilise un tied-weight decoder
(`W^T · hidden`) qui doit rester interne à `SarlQNetwork` (W est partagé
avec l'encoder). Extraire la soustraction casserait l'architecture.

**Quand unifier** : post-Phase F (reproduction validée). Voie :
`SecondOrderCore(dropout, wager_layer, output_mode)` + 2 thin wrappers.

### 🚨 Bug `n_wager_units=2` softmax (review C.4-F1, fixed)

Le port initial avait `torch.softmax` pour `n_wager_units=2`. Faux —
paper attend raw logits. **Fixed en Sprint-08 C.6**. Le port actuel
retourne raw logits quand `n_wager_units=2`.

## 5. Questions ouvertes

- **Q1 :** Pourquoi paper §2.2 mentionne hidden layer Pasquali mais
  student code ne l'a pas implémenté ? Bug d'oubli, ou choix design
  délibéré qui n'a pas été documenté ? L'absence de hidden layer change
  significativement le résultat (Blindsight wager 0.67 → 0.82). Difficile
  à croire qu'un choix avec un tel impact ait été silencieux.

- **Q2 :** Le no-op cascade sur SARL Q-net (pas de dropout) — est-ce
  un side-effect involontaire, ou bien Juan supposait que le 1st-order
  cascade aurait un effet via une régularisation implicite que je ne vois
  pas ? On a noté que paper Table 6 Setting 2 ≠ Setting 1 (Seaquest
  z=-2.59) malgré mathématiquement identiques.

- **Q3 :** Le `retain_graph=True` à travers 50 cascade steps est coûteux
  en mémoire. Est-ce vraiment nécessaire (peut-on détacher `prev_comparison`
  périodiquement) ou est-ce paper-faithful obligatoire ? Test : runs
  avec `detach()` tous les 10 steps vs full unroll.

- **Q4 :** *(à toi, Rémy)* — qu'est-ce qui te paraît non-obvious en
  lisant ce module ?

## 6. Pour le rewrite (Sprint 11+)

### À garder tel quel
- Composition ordre `Comparator → Dropout → cascade → Wager`
- Raw logits pour `n_wager_units=2` (post-C.6 fix)
- `hidden_dim` paramétrable (D.25 D.28)
- Weight init uniforme (0, 0.1)
- Default `dropout=0.5` (Blindsight/AGL)
- Validation shape dans `ComparatorMatrix`

### À améliorer / questions design pour Sprint 11+

- **API du `hidden_dim` paramètre** : `hidden_dim=0` signifie "pas de
  hidden" — sémantique un peu cryptique. Alternative : `hidden_dim:
  int | None = None` (None = pas de hidden, sinon int positif). À
  trancher.
- **Naming `wagering_head` vs `wager`** : le student inline appelle
  `self.wager`, notre port `self.wagering_head`. Cohérence avec
  citations (Koch & Preuschoff "betting") ? À trancher.
- **Docstring shape post-cascade** : être explicite "retourne
  `comparison_out` POST-cascade, à threader en `prev_comparison` au
  step suivant" (C5-fix-3 du review).
- **`ComparatorMatrix` stateless** : envisager fonction libre vs
  `nn.Module` — le Module ne sert qu'à l'extensibilité hypothétique.

### Tests à écrire (avant de toucher au code)
- **Test eq.1 bit-exact** : `ComparatorMatrix(x, y)` ≡ `x - y` pour
  des shapes (B, D) variées
- **Test shape validation** : raise sur shapes différentes
- **Test wager output ranges** : `n_wager_units=1` → sigmoid ∈ [0, 1] ;
  `n_wager_units=2` → raw logits (peut être négatif)
- **Test hidden layer activation** : `hidden_dim=10` ajoute bien
  `Linear(input_dim, 10) → ReLU → Linear(10, n_wager_units)`
- **Test cascade-averaging** : 50 iters avec dropout produisent un
  steady-state proche de `(1-p)·comparison_matrix` (variance reduction
  visible)
- **Test threading prev_comparison** : avec et sans threading, divergence
  numérique attendue

### Opportunités perf identifiées
- **Skipper la cascade en eval mode** (`.eval()` → dropout off → cascade
  = no-op). 50× speedup gratuit. Mais paper-faithful = garder ; à
  discuter (C5-fix-1 du review).
- **`retain_graph` reduction** : détacher `prev_comparison` après
  chaque N steps pour borner la mémoire (pas dans le scope review).

### Connexion avec d'autres modules
- Dépend de `core.cascade.cascade_update`
- Appelé par `domains/blindsight/trainer.py` (via `SecondOrderNetwork`)
- Appelé par `domains/agl/trainer.py` (idem)
- Pas appelé par SARL (utilise `SarlSecondOrderNetwork` séparé)

## Méta — pour une session Claude chat

Prompt prêt à coller dans Claude chat (web) :

```text
J'étudie l'architecture second-order du paper MAPS (Vargas et al. 2025).
Le 2nd-order observe le 1st-order via :

    C_t = X_t - Ŷ_t                          # eq.1, comparator
    C'_t = Dropout(C_t)                       # eq.2
    [cascade sur C'_t pendant 50 iters]      # eq.6 (McClelland 1989)
    W_t = W · C''_t + b                       # eq.3, wager linear
    Loss_BCE = -[y log σ(W) + (1-y) log(1-σ(W))]   # eq.5

Le paper §2.2 cite Pasquali & Cleeremans (2010) "as in" — qui implique
un hidden layer dans le wager. Le code student le supporte en param
mais ne l'utilise jamais. Avec hidden layer, la wager accuracy passe
de 0.67 à 0.82 sur Blindsight.

1. Comment justifier scientifiquement cette différence ? Est-ce un effet
   capacity (plus de params) ou un effet structurel (le ReLU intermédiaire
   change la géométrie des décisions confidence) ?
2. Pasquali 2010 a-t-il publié les détails de cette hidden layer (dims,
   activation) ? Est-ce qu'on retombe pile sur leur archi en mettant
   hidden_dim=100 ?
3. Conceptuellement, qu'est-ce qu'un "comparator + wager" capture par
   rapport à une probabilité de classification standard (output softmax) ?
```
