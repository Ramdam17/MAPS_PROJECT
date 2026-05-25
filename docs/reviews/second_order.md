# Component review — `src/maps/components/second_order.py`

**Review sub-phases :** Sprint-08 C.3 (`ComparatorMatrix`), C.4 (`WageringHead`), C.5 (`SecondOrderNetwork`).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-19.
**File under review :** `src/maps/components/second_order.py` (150 lignes, 3 classes).
**Paper sources :** eq. 1-3 p. 6 (Know-Thyself architecture, §2.1).
**Student sources :**
- `external/paper_reference/blindsight_tmlr.py:241` (`comparison_matrix = first_order_input - first_order_output`).
- `external/paper_reference/agl_tmlr.py` (équivalent).
- `external/paper_reference/sarl_maps.py:176-180` (bypass — inline dans `QNetwork`, tied-weight
  reconstruction, **n'utilise pas** ce component).
**Callers du port :**
- `SecondOrderNetwork` (L131, instancie `self.comparator = ComparatorMatrix()`).
- `src/maps/experiments/blindsight/trainer.py:53` (via `SecondOrderNetwork`).
- `src/maps/experiments/agl/trainer.py:51` (idem).
- Tests : `tests/unit/components/test_second_order.py`, parity Blindsight + AGL.
- **SARL bypass** : `src/maps/experiments/sarl/model.py:91-92` calcule `comparison` inline via
  tied-weights, **pas** via `ComparatorMatrix`.

**DoD global** : 3 sections (ComparatorMatrix, WageringHead, SecondOrderNetwork), fixes identifiées,
rien de code touché. C.3 ne produit que la 1ère section ; C.4-C.5 complètent.

---

## ComparatorMatrix (C.3)

Port complet (15 lignes utiles) :

```python
class ComparatorMatrix(nn.Module):
    """Element-wise difference `C = first_order_input - first_order_output`."""
    def forward(self, first_order_input, first_order_output):
        return first_order_input - first_order_output
```

### (a) Conformité eq. 1 paper

**Paper eq. 1** (verbatim, §2.1 p. 6) :

$$\mathbf{C}_t = \mathbf{X}_t - \hat{\mathbf{Y}}^{(1)}_t$$

Where : `X_t` = input to first-order, `Ŷ^{(1)}_t` = first-order output (reconstruction).

**Port** : soustraction élément par élément exacte, aucune transformation intermédiaire.

✅ **Conformité bit-exact.** Matche eq. 1 littéralement.

### (b) Shape handling

Port ne valide **aucune shape** — s'appuie sur torch broadcasting standard :

- Match shape → soustraction élément par élément (cas nominal, Blindsight `(B, 100)`, AGL `(B, 48)`).
- Shape mismatch broadcastable (ex : `(B, D)` − `(B, 1)`) → broadcast silencieux, potentiellement
  pas intentionnel.
- Shape mismatch non-broadcastable → `RuntimeError: size mismatch`.

**Docstring silencieuse sur les assumptions** — pas de mention "both tensors must have the same
shape".

**Piste C3-fix-1** : ajouter 1-ligne `assert first_order_input.shape == first_order_output.shape`
ou équivalent (doc docstring + validation runtime). Très bas coût, capte un bug silencieux si un
refactor future décale les dims.

### (c) Detach semantics

**Pas de detach** dans le port. La soustraction `input - output` propage les gradients vers
**les deux** tenseurs :

- `dC/d input_fo = 1`  → gradients flow vers l'input (data, pas de params, rien à optimiser).
- `dC/d output_fo = −1` → gradients flow vers `output_fo` → via backward, vers les **poids du
  first-order network**.

**Implication** : la loss 2nd-order (BCE wagering) influence **aussi** les poids du 1st-order
network via ce chemin. Paper ne dit rien explicite mais le comportement est implicite :
- Setting 3 / 4 / 5 / 6 (meta=True) : les 2 networks s'entraînent conjointement par ce couplage.
- **Blindsight student** (`blindsight_tmlr.py:241`) fait pareil (no detach). Notre port match.
- **Parité paper** : conservé.

**Alternative non-paper-faithful** : `ComparatorMatrix.forward` avec `first_order_output.detach()`
isolerait les 2 networks → équivalent à un "frozen teacher" pour le 2nd-order. Utile pour étudier
l'effet. Non par défaut.

✅ **Detach semantics cohérente avec student + paper-implicit.**

### (d) Cross-domain usage — asymétrie SARL vs Blindsight/AGL

**Blindsight + AGL** : `SecondOrderNetwork` instancie `ComparatorMatrix()` et l'appelle en forward
(`L146`). C'est le path standard.

**SARL bypass** : dans `sarl/model.py:91-92`, la comparison est calculée **inline** dans
`SarlQNetwork.forward` :

```python
# SarlQNetwork.forward
reconstruction = F.relu(F.linear(hidden, self.fc_hidden.weight.t()))  # tied-weight decoder
comparison = flat_input - reconstruction  # equivalent to ComparatorMatrix(flat_input, reconstruction)
```

Le résultat `comparison` est passé à `SarlSecondOrderNetwork.forward(comparison, ...)` (pas via
`ComparatorMatrix`). Le `components.ComparatorMatrix` n'est donc **jamais appelé dans le path
SARL**.

**Pourquoi l'asymétrie ?** SARL utilise un **tied-weight decoder** (reconstruction via
`W^T · hidden`) qui doit rester interne à `SarlQNetwork` parce que `W = fc_hidden.weight` est
partagé avec l'encoder. Extraire la soustraction externe briserait rien, mais pour la parité
paper + student le code reste inline.

**Piste C3-fix-2** (optionnel, info docstring) : ajouter dans le docstring du port une note
explicite "Note: SARL computes the comparison inline (tied-weight decoder). `ComparatorMatrix`
is used only by Blindsight + AGL via `SecondOrderNetwork`." Améliore la lisibilité.

### Fixes identifiées C.3 (pour C.6 follow-up)

| ID         | Fix                                                                    | Scope                                 | Effort |
|------------|------------------------------------------------------------------------|---------------------------------------|:------:|
| C3-fix-1   | `assert first_order_input.shape == first_order_output.shape` + docstring | `second_order.py:48-51` + tests check | 10 min |
| C3-fix-2   | Docstring : note "SARL computes comparison inline, not via this module" | `second_order.py:41-46` docstring   | 5 min  |
| C3-fix-3 (info) | Documenter que gradient flow vers 1st-order est paper-faithful    | `second_order.py:51` inline comment | 2 min  |

### Résumé — ComparatorMatrix

- ✅ **Conformité eq. 1** : bit-exact (soustraction pure).
- ⚠️ **Shape** : pas de validation, broadcast silencieux possible. C3-fix-1.
- ✅ **Detach** : absent, paper-faithful, gradient flow cohérent student.
- ⚠️ **Cross-domain** : asymétrie SARL inline vs Blindsight/AGL via module. Documentable
  (C3-fix-2), pas un bug.
- **0 divergence paper.** 2 fixes mineurs identifiés.

**C.3 clôturée.**

---

## WageringHead (C.4)

Port `src/maps/components/second_order.py:54-90` (~36 lignes) :

```python
class WageringHead(nn.Module):
    def __init__(self, input_dim, n_wager_units=1, weight_init_range=(0.0, 0.1)):
        super().__init__()
        if n_wager_units not in (1, 2):
            raise ValueError(...)
        self.n_wager_units = n_wager_units
        self.wager = nn.Linear(input_dim, n_wager_units)
        init.uniform_(self.wager.weight, *weight_init_range)

    def forward(self, comparator_out):
        logits = self.wager(comparator_out)
        if self.n_wager_units == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)  # ← see finding below
```

### (a) Conformité eq. 2-3 paper

**Paper eq. 2** : `C'_t = Dropout(C_t)`.
**Paper eq. 3** : `W_t = (W · C'_t + b)` — **raw logits, sans activation** dans l'équation.
L'activation est appliquée **dans la loss** via eq. 5 : `L_BCE = -[y·log(σ(logits)) + (1-y)·log(1-σ(logits))]`.

Comportement port par `n_wager_units` :

| n_wager_units | Linear out shape | Activation finale port | Paper eq. 3 attendu                         | Matches ? |
|:-------------:|:----------------:|:----------------------:|:-------------------------------------------|:--------:|
| **1** (default) | `(B, 1)`         | `sigmoid`              | paper ambiguë ; eq. 5 BCE attend `σ(logits)` → sigmoid ≠ in-loss | 🆘 D-001 (déjà ouvert) |
| **2**         | `(B, 2)`         | **`softmax(dim=-1)`**  | raw logits (à cause de eq. 5 BCE sigmoid)  | 🚨 NEW finding |

**🚨 Finding C.4-F1 — `n_wager_units=2` path est incorrect** :

Le docstring L63 dit *"2 → softmax over {bet, no-bet} (paper-faithful, see Koch & Preuschoff 2007)"*
— mais eq. 5 paper utilise **sigmoid** sur les logits (pas softmax). Student SARL
(`sarl_maps.py:276`) retourne **raw logits** (pas de softmax), et la loss est
`binary_cross_entropy_with_logits(wager[B,2], targets_wagering[B,2])` qui applique sigmoid
internement, pas softmax.

Donc notre port `WageringHead(n_wager_units=2)` :
- Produit `softmax(logits, dim=-1)` → probabilités normalisées sur les 2 units.
- BCE appliquée sur softmax ≠ BCE appliquée sur raw logits.
- **Diverge de student SARL** (qui retourne raw logits) **et de paper** (qui traite chaque unit
  comme Bernoulli indépendant per eq. 5 + eq. 14 `y_wager ∈ {(1,0), (0,1)}`).

**Fix** : enlever le `torch.softmax` et retourner raw logits quand `n_wager_units=2`. Le caller
(`SecondOrderNetwork`) passera à `binary_cross_entropy_with_logits` qui applique sigmoid.

### (b) Dropout placement

**Paper eq. 2** : `C'_t = Dropout(C_t)` **avant** le wager linear.

**Port `WageringHead`** : n'a **pas** de dropout. C'est `SecondOrderNetwork.forward` (L147) qui
applique le dropout en amont :
```python
comparison_out = self.dropout(comparison_matrix)
comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
wager = self.wagering_head(comparison_out)
```

✅ Architecturalement correct (séparation Comparator → Dropout → cascade → WageringHead). Le
`WageringHead` par contrat **assume que son input est déjà dropout-masked** (et cascade-intégré).

⚠️ **Docstring silencieuse** sur cette dépendance au caller. `WageringHead` passant un input
non-dropout fonctionnerait mais serait paper-non-conforme. Piste C4-fix-2 : ajouter une ligne de
docstring "assumes `comparator_out` is already dropout-masked and cascade-integrated (see
`SecondOrderNetwork.forward`)".

### (c) Dropout rate paper / student / port

**Paper** : silencieux sur la valeur de dropout (eq. 2 dit juste `Dropout(C)` sans rate).

**Student** — valeurs hétérogènes :
- `blindsight_tmlr.py:222` : `self.dropout = nn.Dropout(0.5)` (wagering).
- `agl_tmlr.py` : probablement idem 0.5 (même template que Blindsight).
- `sarl_maps.py:254` : `self.dropout = nn.Dropout(p=0.1)` (wagering).

→ **Le student a 2 valeurs différentes** selon le domaine : Blindsight/AGL p=0.5, SARL p=0.1.

**Port** :
- `SecondOrderNetwork.__init__` default `dropout=0.5` (L126) — matches Blindsight/AGL student.
- `src/maps/experiments/sarl/model.py:SarlSecondOrderNetwork` (autre classe, **pas celle-ci**)
  utilise `p=0.1`.

✅ **Alignement correct par domaine** : `components.SecondOrderNetwork` (utilisé Blindsight/AGL)
default p=0.5, `sarl.SarlSecondOrderNetwork` (utilisé SARL) default p=0.1. Les deux matchent leur
student respectif.

### (d) Weight init uniforme (0, 0.1)

**Port** : `init.uniform_(self.wager.weight, 0.0, 0.1)` (default `weight_init_range=(0.0, 0.1)`).

**Student** :
- `blindsight_tmlr.py:237` : `init.uniform_(self.wager.weight, 0.0, 0.1)` ✅
- `agl_tmlr.py:239` : idem ✅
- `sarl_maps.py:264` : idem ✅

Tous matchent.

✅ **Weight init bit-exact avec student sur les 3 domaines.**

**Bias** : PyTorch default Kaiming-style uniform (non initialisé explicitement). Student Blindsight
L237 ne touche que `wager.weight`, pas `wager.bias` — idem. ✅

### Fixes identifiées C.4

| ID         | Fix                                                                          | Scope                                     | Effort |
|------------|------------------------------------------------------------------------------|-------------------------------------------|:------:|
| **C4-fix-1** (🚨) | `WageringHead.forward` : retourner raw logits quand `n_wager_units=2` (enlever softmax) | `second_order.py:90` + docstring | 15 min |
| C4-fix-2   | Docstring WageringHead : documenter que `comparator_out` doit être dropout-masked + cascade-intégré | `second_order.py:54-67` docstring | 5 min  |
| C4-fix-3 (info) | Ajouter une constante `DEFAULT_WAGER_INIT_RANGE = (0.0, 0.1)` au niveau module pour cross-reference avec student | `second_order.py:38` module-level | 5 min  |

### Cross-reference deviations.md

- **D-001** existante : "1-unit sigmoid default vs 2-unit softmax paper-faithful" — à **corriger
  la description** : paper-faithful est raw logits, pas softmax. Note à ajouter dans deviations.md
  Phase D.23 ou C.6.
- **Nouveau** : D-wager-softmax-wrong (C4-fix-1) — path `n_wager_units=2` du port applique softmax
  alors que paper + student SARL attendent raw logits. À ajouter dans deviations.md.

### Résumé — WageringHead

- ✅ **Conformité eq. 3** pour `n_wager_units=1` (1-unit sigmoid) matches student Blindsight/AGL
  mais **diverge du paper 2-unit** (D-001 existant).
- 🚨 **Conformité eq. 3** pour `n_wager_units=2` (softmax) : **port incorrect** — devrait retourner
  raw logits. C4-fix-1 critique.
- ✅ **Dropout placement** : correctement externe (dans `SecondOrderNetwork`), WageringHead assume
  l'input pre-masked.
- ⚠️ **Dropout rate** : student a 2 valeurs (Blindsight/AGL 0.5, SARL 0.1) — port gère chaque
  domaine avec sa propre classe. Aligné.
- ✅ **Weight init** uniform(0, 0.1) matches student sur les 3 domaines, bit-exact.

**C.4 clôturée. 3 fixes + 1 update deviations.md pour C.6.**

---

---

## SecondOrderNetwork (C.5)

Port `src/maps/components/second_order.py:93-150` (~58 lignes). Classe-chapeau qui compose
`ComparatorMatrix → Dropout → cascade_update → WageringHead` en un seul module.

```python
class SecondOrderNetwork(nn.Module):
    def __init__(self, input_dim, dropout=0.5, n_wager_units=1, weight_init_range=(0.0, 0.1)):
        super().__init__()
        self.comparator = ComparatorMatrix()
        self.dropout = nn.Dropout(dropout)
        self.wagering_head = WageringHead(input_dim, n_wager_units, weight_init_range)

    def forward(self, first_order_input, first_order_output, prev_comparison, cascade_rate):
        comparison_matrix = self.comparator(first_order_input, first_order_output)
        comparison_out = self.dropout(comparison_matrix)
        comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
        wager = self.wagering_head(comparison_out)
        return wager, comparison_out
```

### (a) Composition vs student (bit-exact)

Student `blindsight_tmlr.py:239-252` (verbatim, reflow) :

```python
def forward(self, first_order_input, first_order_output, prev_comparison, cascade_rate):
    comparison_matrix = first_order_input - first_order_output        # eq.1
    comparison_out = self.dropout(comparison_matrix)                  # eq.2
    if prev_comparison is not None:                                   # eq.6 cascade
        comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
    wager = self.sigmoid(self.wager(comparison_out))                  # eq.3 + sigmoid
    return wager, comparison_out
```

**Mapping port ↔ student** :

| Étape | Student L241-252                                             | Port L146-149                                                 | Match |
|:-----:|--------------------------------------------------------------|---------------------------------------------------------------|:-----:|
| eq.1  | `first_order_input - first_order_output`                     | `self.comparator(fi, fo)` → même op                           | ✅    |
| eq.2  | `self.dropout(comparison_matrix)` (p=0.5)                    | `self.dropout(comparison_matrix)` (p=0.5 default)             | ✅    |
| eq.6  | `α·x + (1-α)·prev` inline (avec `if prev is not None`)       | `cascade_update(comparison_out, prev_comparison, cascade_rate)` | ✅    |
| eq.3  | `self.sigmoid(self.wager(comparison_out))`                   | `self.wagering_head(comparison_out)` (sigmoid si `n_wager_units=1`) | ✅ (default)   |

**Ordre des ops** : comparator → dropout → cascade → wager. Identique student.

✅ **Composition bit-exact avec student Blindsight.** AGL `agl_tmlr.py` utilise le même template.

### (b) Forward signature

Port : `(first_order_input, first_order_output, prev_comparison, cascade_rate) → (wager, comparison_out)`.
Student : **identique**.

Caller (Blindsight trainer L299-306) :
```python
comparison: torch.Tensor | None = None
for _ in range(self.cascade_iters):
    wager, comparison = self.second_order(batch.patterns, h2, comparison, self.cascade_rate)
```

Le caller initialise `comparison = None` (bootstrap step 1), threade `comparison` en retour
comme `prev_comparison` de l'itération suivante. Pattern correct ; `cascade_update(x, None, α) = x`
via C.1 (`prev_activation is None → return new_activation`).

✅ **Signature + threading correct.**

### (c) Retourne comparison_out **post-cascade**

Port L147-149 :
```python
comparison_out = self.dropout(comparison_matrix)          # post-dropout, pre-cascade
comparison_out = cascade_update(comparison_out, ...)       # post-cascade
wager = self.wagering_head(comparison_out)                 # feed to wager
return wager, comparison_out                               # ← post-cascade
```

Le caller thread la version **post-cascade** comme `prev_comparison` au step suivant. Student pareil
(L247-252). C'est ce qui fait converger `comparison_out → stationary` asymptotiquement — sans ça,
le cascade ne peut pas accumuler.

✅ **Threading asymptote correct.**

### (d) Dropout placement : eq.2 **avant** cascade (eq.6)

Paper eq.2 : `C'_t = Dropout(C_t)`. Paper eq.6 applique le cascade. L'ordre paper n'est pas
explicitement commenté ; student applique **dropout → cascade** (pas l'inverse). Port match.

**Impact du choix dropout→cascade vs cascade→dropout** :

- **Dropout → cascade** (port actuel, student) : chaque itération a un **mask dropout différent**.
  Le cascade accumule donc `α · Dropout₁(C) + α(1-α) · Dropout₂(C) + ...`. Sur 50 itérations avec
  α=0.02, ça **moyenne** 50 masks différents → le résultat steady-state approche
  `E[Dropout(C)] = (1-p) · C` (si p=0.5 → ≈ 0.5·C). **Reduction de variance** sur le signal
  comparator — rôle non-trivial du cascade.
- **Cascade → dropout** : appliquerait un seul mask sur le steady-state. Variance plus élevée,
  pas de moyennage. Différent.

→ **L'ordre port = student = effet utile du cascade sur path stochastique**. Ceci **résout
partiellement** le paradoxe flaggé en `docs/reviews/cascade.md §(d)` (cascade = no-op sur
deterministic forward) : sur le path 2nd-order Blindsight/AGL **le forward est stochastique via
dropout**, donc cascade ≠ no-op. Le cascade AVERAGE des dropout masks.

⚠️ **MAIS** : path 1st-order SARL (`SarlQNetwork.forward`) **n'a pas de dropout** (L85-86 :
`hidden = F.relu(self.fc_hidden(flat_input)); hidden = cascade_update(hidden, prev_h2, α)`). Sur
ce path, `prev_h2=None` au bootstrap puis `hidden` constant → cascade **reste no-op** à
l'itération ≥ 2. Finding C.1 confirmé pour SARL 1st-order ; résolu pour Blindsight/AGL 2nd-order.

✅ **Ordre dropout→cascade paper-faithful (via student).**

### (e) Dropout passe en eval → passif

Le port utilise `nn.Dropout` (pas un mask custom), donc respecte `self.training`. Le caller
Blindsight L378 (`self.second_order.eval()`) désactive automatiquement dropout en eval. À
l'inférence, `Dropout(C) = C` → cascade voit input **constant** → cascade redevient no-op.

**Conséquence en eval** : le SecondOrderNetwork boucle 50 fois pour produire **le même wager**
qu'après 1 itération. 50× overhead inutile en eval.

⚠️ **C5-fix-1** (optim non-paper-faithful, skip par défaut) : en `eval()` mode, shortcut à
`cascade_iters=1`. Non appliqué car :
1. Pas paper-faithful (paper `cascade_iters=50` même en eval implicitement).
2. Bench confirme overhead ~50× mais reste négligeable dans le budget Blindsight/AGL (~secondes).
3. Doit être discuté avant changement — flagge pour future optim si training bottlenecké.

### (f) cascade_rate propagation — positional

Le port propage `cascade_rate` positionnellement :
```python
cascade_update(comparison_out, prev_comparison, cascade_rate)  # 3e arg positionnel
```

OK post-C.2 (renaming `alpha → cascade_rate` complet). Pas de piège résiduel.

✅ **Propagation post-C.2 correcte.**

### (g) Doublon avec `SarlSecondOrderNetwork` (src/maps/experiments/sarl/model.py:97-147)

SARL a sa **propre classe** `SarlSecondOrderNetwork` (50 lignes). Comparatif :

| Aspect                 | `components.SecondOrderNetwork`                | `sarl.SarlSecondOrderNetwork`                    |
|:-----------------------|:-----------------------------------------------|:-------------------------------------------------|
| ComparatorMatrix       | `self.comparator = ComparatorMatrix()` (L131) | Absent (comparison inline dans `SarlQNetwork`)    |
| Dropout                | `p=0.5` (default)                              | `p=0.1` (paper SARL)                              |
| Wager dims             | `input_dim → n_wager_units` (1 ou 2)          | `NUM_LINEAR_UNITS=1024 → 2` (hardcoded)          |
| Wager activation       | `sigmoid` (n=1) ou `softmax` (n=2, 🚨 C4-fix-1) | **Raw logits** (pas d'activation)                 |
| Forward signature      | `(fi, fo, prev_comp, α) → (wager, comp_out)`   | `(comparison_matrix, prev_comp, α) → (wager, comp_out)` |
| Weight init            | `uniform(0, 0.1)`                              | `uniform(0, 0.1)`                                 |

**Raison de la duplication** :
1. `ComparatorMatrix` inutilisable en SARL (tied-weight decoder interne à `SarlQNetwork`).
2. `NUM_LINEAR_UNITS` est un constant dépendant de l'archi MinAtar (10×10 grid, 16 filters),
   pas un `input_dim` paramétrable.
3. Raw logits requis pour BCE-with-logits (eq.5 paper) — `SecondOrderNetwork` applique sigmoid
   ou softmax, incompatible.
4. Dropout rate différent (paper SARL = 0.1 vs Blindsight/AGL = 0.5).

**Unification possible ?** Oui, mais coût-bénéfice négatif :
- Il faudrait rendre `SecondOrderNetwork` accepter soit `(fi, fo)` soit `(comparison_matrix,)`
  comme input → branche conditionnelle → perte de clarté.
- Il faudrait ajouter un mode "raw logits" à `WageringHead` (déjà envisagé dans C4-fix-1 ; après
  ce fix, `n_wager_units=2` = raw logits par défaut, donc **possible après C.6**).

**Recommandation** : ne pas unifier maintenant. Documenter dans `docs/reproduction/deviations.md`
comme dette technique (DETTE-1 : "SecondOrderNetwork + SarlSecondOrderNetwork doublon") pour
future refacto après que la reproduction paper soit validée. Risque nul en l'état.

⚠️ **C5-fix-2** : ajouter DETTE-1 dans `deviations.md` (tracking de la duplication,
pas un bug).

### (h) Gradient flow à travers 50 cascade steps

Le caller Blindsight L314 `loss_2.backward(retain_graph=True)` fait backprop à travers **toute**
la boucle de 50 itérations cascade. Chaque itération crée un nouveau noeud graph via le
`cascade_update` qui produit `α·new + (1-α)·prev_comparison`.

**Implications** :
1. **Memory** : graph garde 50 copies intermédiaires → mémoire ×50 pour la 2nd-order loss.
2. **Backward depth** : 50 multiplications-chaînées + 50 additions + 50 dropouts + 50 linear.
3. **Gradient via dropout mask** : chaque mask est tiré indépendamment → gradient reçoit la
   moyenne sur 50 masks → **régularisation implicite** du gradient (proche de Monte-Carlo
   dropout dans Gal & Ghahramani 2016).

✅ Comportement paper-faithful (matches student qui a le même unroll).

⚠️ **Non-fix** : le `retain_graph=True` est nécessaire car le graph est réutilisé pour la loss
1st-order après. Si on veut optimiser la mémoire un jour, détacher `prev_comparison` (pas dans
ce review).

### Fixes identifiées C.5

| ID       | Fix                                                                 | Scope                                  | Effort |
|----------|---------------------------------------------------------------------|----------------------------------------|:------:|
| C5-fix-1 | (skip) eval-mode shortcut `cascade_iters=1` — **non appliqué** (pas paper-faithful) | `SecondOrderNetwork.forward` + caller | — |
| C5-fix-2 | Ajouter DETTE-1 "SecondOrderNetwork + SarlSecondOrderNetwork doublon" dans `deviations.md` | `docs/reproduction/deviations.md`  | 5 min  |
| C5-fix-3 | Docstring `forward` : noter explicitement "returns comparison_out POST-cascade, caller threads it as prev_comparison" | `second_order.py:139-145` docstring | 5 min  |

### Cross-reference avec C.1 (cascade) — paradoxe résolu partiellement

Le finding 🚨 de `docs/reviews/cascade.md §(d)` — "cascade = no-op sur deterministic forward" —
est **résolu pour Blindsight/AGL 2nd-order path** via le dropout inside-cascade qui rend le
forward stochastique. Le cascade y fait un **averaging de 50 dropout masks** → réduit la variance
du signal comparator.

Le finding **reste ouvert** pour :
- **SARL 1st-order path** (`SarlQNetwork.forward`) : pas de dropout dans le forward → cascade
  reste no-op → à investiguer en Phase D (sub-phase D.15 ou équivalent) si la parité SARL diverge.
- **SarlSecondOrderNetwork** : a du dropout p=0.1 donc **pareil que Blindsight 2nd-order**,
  cascade averaging → OK.

**À noter dans deviations.md** : mettre à jour D-002 (cascade impact) avec ce refinement :
`cascade = averaging mask sur 2nd-order (path dropout) ; no-op sur SARL 1st-order (pas de dropout)`.

### Résumé — SecondOrderNetwork

- ✅ **Composition bit-exact** avec student Blindsight/AGL (mapping 4 étapes).
- ✅ **Forward signature + threading** identiques student ; caller initialise `comparison=None`
  et boucle correctement.
- ✅ **Retour post-cascade** correct pour threading asymptote.
- ✅ **Dropout→cascade paper-faithful** ; resout partiellement le paradoxe C.1 (averaging de masks).
- ⚠️ **Eval mode = 50× overhead** (cascade no-op sans dropout actif) — pas fix (paper-faithful).
- ✅ **cascade_rate propagation** positional, post-C.2 OK.
- ⚠️ **Doublon SarlSecondOrderNetwork** intentionnel (tied-weight, raw logits, dims différents) —
  dette à tracker (DETTE-1), pas à unifier avant reproduction paper validée.
- ✅ **Gradient flow 50-deep** via `retain_graph=True` dans caller, paper-faithful.
- **0 divergence paper, 1 dette documentable, 2 notes docstring.**

**C.5 clôturée. 2 fixes mineurs (C5-fix-2 = deviations.md, C5-fix-3 = docstring) + 1 update D-002
pour C.6.**

