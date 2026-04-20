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

## WageringHead (C.4) — à remplir

*(placeholder — sera rempli en sub-phase C.4)*

---

## SecondOrderNetwork (C.5) — à remplir

*(placeholder — sera rempli en sub-phase C.5)*
