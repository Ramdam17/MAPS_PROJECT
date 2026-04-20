# Component review — `src/maps/experiments/sarl_cl/model.py`

**Review sub-phase :** Sprint-08 D.17.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl_cl/model.py` (257 L, 3 classes + 1 helper).
**Paper sources :** §2.3 p.9 (Continual Learning architecture), eq.12 (reconstruction), D-sarl_cl-*
deviations.
**Student sources :** `external/paper_reference/sarl_cl_maps.py:117-219` (QNetwork,
AdaptiveQNetwork, SecondOrderNetwork).

**Callers du port :**
- `sarl_cl/training_loop.py:_build_networks` (policy/target/second/teachers instantiation).

**DoD D.17** : doc créé, 3 classes + 1 helper audités, 1 deviation resolved, 1 queued D.20, 0 code touché.

---

## (a) `_size_linear_unit` + `NUM_LINEAR_UNITS`

Idem SARL standard : `NUM_LINEAR_UNITS = 8 * 8 * 16 = 1024`. Bit-parity.

✅ **Trivial, aucune divergence.**

## (b) `SarlCLQNetwork` vs `SarlQNetwork` (SARL standard)

### Différences architecturales key

| Aspect                   | SARL (tied)                          | SARL+CL (this class)                  |
|:-------------------------|:-------------------------------------|:---------------------------------------|
| Decoder                  | tied: `fc_hidden.weight.T`           | **explicit `fc_output = Linear(128, 1024, bias=True)`** |
| Cascade applied to       | `Hidden` (128-dim)                   | **`Output` (1024-dim)**                |
| `actions` input dim      | 128 (Hidden)                         | **1024 (Output)**                      |
| Comparison               | `Input − ReLU(W^T·Hidden + b_recon)` | `Input − Output` (cascade-integrated) |

### (b1) 🆗 D-sarl-recon-bias **ne s'applique pas** à SARL+CL

Paper eq.12 `ReLU(W^T·Hidden + b_recon)` est la forme **tied-weight** — spécifique à SARL standard
(où le decoder réutilise `fc_hidden.weight.T`). Dans SARL+CL, le decoder `fc_output` est un
`nn.Linear(128, 1024)` séparé avec `bias=True` par défaut → il a **déjà** son bias learnable
implicitement.

→ **Le fix D-sarl-recon-bias (D.7) ne nécessite pas de contrepartie SARL+CL.** ✅

### (b2) Cascade position — Hidden vs Output

SARL intègre cascade sur `Hidden` (128-dim) ; SARL+CL sur `Output` (1024-dim). Paper p.9 silent
sur cette distinction. Student `sarl_cl_maps.py:151` confirme cascade sur Output. ✅ **Port =
student bit-parity**.

### (b3) Mapping port ↔ student

Student `sarl_cl_maps.py:117-158` :
```python
self.conv = nn.Conv2d(in_channels, 16, 3, 1)
self.sigmoid = nn.Sigmoid()          # unused artifact
self.fc_hidden = nn.Linear(num_linear_units, 128)
self.fc_output = nn.Linear(128, num_linear_units)
self.actions = nn.Linear(num_linear_units, num_actions)

# forward:
x = f.relu(self.conv(x))
Input = x.view(x.size(0), -1)
Hidden = f.relu(self.fc_hidden(Input))
Output = f.relu(self.fc_output(Hidden))
if prev_h2 is not None:
    Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2
x = self.actions(Output)
Comparisson = Input - Output
return x, Hidden, Comparisson, Output
```

Port L90-117 **identique structurellement** — même ordre init (conv → sigmoid → fc_hidden →
fc_output → actions), même forward avec `cascade_update` au lieu de l'inline cascade. Bit-parity
preserved.

✅ **Bit-parity student** sur init RNG + forward pipeline.

### (b4) `self.sigmoid` unused — parity artifact

Port L97 garde `self.sigmoid = nn.Sigmoid()` jamais appelé dans forward. Student L127. Pas de
conséquence RNG (nn.Sigmoid sans params). ✅ **Parity preserved**.

## (c) `SarlCLSecondOrderNetwork` vs `SarlSecondOrderNetwork`

### Différences architecturales key

| Aspect                     | SARL standard                    | SARL+CL (this class)                 |
|:---------------------------|:---------------------------------|:-------------------------------------|
| Pre-wager layer            | (none)                           | **`comparison_layer = Linear(1024, 1024)`** |
| Pre-wager non-linearity    | (none)                           | **`F.relu(comparison_layer(C))`** + dropout |
| Cascade applied to         | `C` (comparison direct)          | **post-`ReLU(comparison_layer(C))`** |
| Wager layer                | `Linear(NUM_LINEAR_UNITS, 2)`    | `Linear(NUM_LINEAR_UNITS, 2)` idem  |
| `comparison_layer` init    | N/A                              | `uniform(-1, 1)` (paper L239)        |
| `wager` init               | `uniform(0, 0.1)` (paper L239)   | `uniform(0, 0.1)` idem               |
| Output                     | raw logits (B, 2)                | raw logits (B, 2) idem               |

### (c1) 🚨 DETTE-1 étendue — **triple duplication SecondOrder**

Rappel C.6 : DETTE-1 = doublon `components.SecondOrderNetwork` + `sarl.SarlSecondOrderNetwork`.
D.17 confirme **triple duplication** :
1. `components.SecondOrderNetwork` (Blindsight/AGL, tied-friendly, optional n_wager_units)
2. `sarl.SarlSecondOrderNetwork` (SARL tied-weight path, fixed 2-unit raw logits)
3. `sarl_cl.SarlCLSecondOrderNetwork` (CL explicit decoder + comparison_layer + dropout position différent)

**Justification** : les 3 variants diffèrent **architecturalement** (comparison_layer présent
seulement dans CL, dropout position, wager input dim) et **paramétriquement**. Unifier demanderait
~5 branches conditionnelles → lecture hostile.

**Piste D17-fix-1** : mettre à jour DETTE-1 dans `deviations.md` pour ajouter le 3e variant.
**Pas d'unification avant Phase F validation**.

### (c2) `comparison_layer` position paper-faithful

Paper §2.3 silent sur la présence d'une `comparison_layer` dédiée. Student `sarl_cl_maps.py:221+`
(`SecondOrderNetwork` CL class) en a une. Port preserve → architectural parity student ✅.

**Note** : cette couche supplémentaire explique pourquoi l'architecture CL est plus riche que
SARL simple. Paper Tables 5/6 CL colonnes peuvent refléter cet avantage.

### (c3) Dropout position — "inside" la cascade ?

Port L173-174 :
```python
comparison_out = self.dropout(F.relu(self.comparison_layer(comparison_matrix)))
comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
wager = self.wager(comparison_out)
```

Dropout appliqué **AVANT** cascade (= même pattern que `components.SecondOrderNetwork` Blindsight/AGL,
pas comme SARL standard qui fait dropout après). Paper eq.2 `C' = Dropout(C)` **avant** cascade
eq.6 → **paper-faithful**.

✅ **Cohérent paper eq.2**.

### (c4) `_in_channels` stored mais unused

Port L150 `self._in_channels = in_channels  # API parity; unused in forward`. Student
constructor `def __init__(self, in_channels):`. Port preserve signature pour parity. ✅

## (d) `AdaptiveQNetwork` — cross-game channel adapter

### (d1) ✅ D-sarl_cl-channel-adapter **RESOLVED** — paper-faithful

Paper §2.3 p.9 prescribes : *"a 1×1 convolution + ReLU input adapter"*.

Port L206-209 :
```python
self.input_adapter = nn.Sequential(
    nn.Conv2d(max_input_channels, max_input_channels, kernel_size=1, stride=1),
    nn.ReLU(),
)
```

Student `sarl_cl_maps.py:168-171` idem.

→ **Paper-faithful confirmed**. D-sarl_cl-channel-adapter entry status "(to verify)" → **✅
resolved in-place** (no fix needed, port = paper = student).

**Piste D17-fix-2** : update `deviations.md` D-sarl_cl-channel-adapter status to ✅ resolved.

### (d2) ❌ D-sarl_cl-max-channels still OPEN

Paper Table 11 CL row : `max_input_channels = 10`. Port config `sarl_cl.yaml:27` + dataclass
default = **7**. Port *code* est paramétrable (init arg), seul le default diverge.

Impact : Seaquest a 10 channels → avec `max=7`, **Seaquest est tronqué à 7** (channels dropped).
Paper curriculum utilise Seaquest full.

→ **Still ❌**, fix queued **D.20**.

### (d3) Zero-padding logic (`adapt_input` L228-240)

```python
if x.size(1) < self.max_input_channels:
    padding = torch.zeros(x.size(0), max - x.size(1), x.size(2), x.size(3), ...)
    x = torch.cat([x, padding], dim=1)
```

Student idem L191-200. Pad **à droite** (channels > current). Paper silent sur la direction —
arbitrary convention, student parity preserved.

⚠️ Note : `adapt_input` n'est appelé que si `x.size(1) < max`. Si games ont **plus** de channels
que max (ex: Seaquest 10 avec max=7 port), aucun truncation n'est fait — le forward va échouer
avec RuntimeError mismatch. **Seaquest crash silent si max=7**. Fix via D.20 (set max=10).

### (d4) `_get_conv_output_size` dynamic probe

Port L221-226 probe avec un dummy input pour compute flat conv output size. Student idem
(`numpy.prod(output.size()[1:])`). Élégant — si kernel/stride change, la taille s'adapte
automatiquement. ✅ **Robust vs hardcoded 1024**.

### (d5) Dead attributes ? No — `AdaptiveQNetwork` utilise ses layers

Contrairement à `SarlCLQNetwork.sigmoid` unused, l'`AdaptiveQNetwork` a tous ses attributes
utilisés dans forward. ✅ Clean.

## (e) Cross-reference `load_partial_state_dict`

**Not defined in this file** — probablement dans `training_loop.py` ou trainer. `grep` pour
localiser.

(Outside D.17 scope — sera reviewed D.19 `sarl_cl/training_loop.py`.)

## Fixes identifiées D.17

| ID        | Fix                                                                         | Scope                            | Effort |
|:----------|-----------------------------------------------------------------------------|----------------------------------|:------:|
| D17-fix-1 | Update DETTE-1 entry dans `deviations.md` — note 3rd variant `SarlCLSecondOrderNetwork` | `deviations.md`             | 5 min  |
| D17-fix-2 | Mark D-sarl_cl-channel-adapter ✅ resolved (paper-faithful confirmed)       | `deviations.md`                   | 2 min  |
| D17→D.20  | D-sarl_cl-max-channels 7→10 (config + dataclass default)                   | queued D.20                      | queued |

## Cross-reference deviations.md

- **D-sarl_cl-channel-adapter** (⚠️ "to verify") → **✅ resolved D.17** via D17-fix-2.
- **D-sarl_cl-max-channels** (❌) → still queued D.20 (config fix).
- **DETTE-1** : étendue de doublon → triplon. D17-fix-1 update.
- **D-sarl-recon-bias** (D.7 resolved) : **n'affecte pas** SARL+CL (decoder déjà explicit bias). ✅

## Résumé — `sarl_cl/model.py`

- ✅ **`_size_linear_unit` + NUM_LINEAR_UNITS** bit-parity SARL.
- ✅ **`SarlCLQNetwork`** bit-parity student (conv + fc_hidden + explicit fc_output + actions),
  cascade sur Output (1024), paper-distinct de SARL tied-weight.
- ✅ **D-sarl-recon-bias ne s'applique pas** (decoder explicit Linear a déjà bias learnable).
- ✅ **`SarlCLSecondOrderNetwork`** architecture paper-faithful avec `comparison_layer` explicit
  + dropout avant cascade (paper eq.2).
- 🚨 **DETTE-1 étendue** au 3e variant — D17-fix-1 update.
- ✅ **`AdaptiveQNetwork` 1×1 conv + ReLU adapter** paper-faithful — **D-sarl_cl-channel-adapter
  resolved**.
- ❌ **`max_input_channels=7` default** vs paper=10 — D-sarl_cl-max-channels queued D.20.
- ✅ **Zero-padding adapt_input** student parity.
- ✅ **Dynamic conv output size** via dummy forward — robust.
- **0 nouvelle deviation, 1 existing resolved (channel-adapter), 1 existing still open
  (max-channels), 1 DETTE extended.**

**D.17 clôturée. 2 fixes `deviations.md` queued (D17-fix-1 DETTE-1 update + D17-fix-2 channel-adapter
resolve). D.20 accumule : weights + num_frames + max-channels = 3 CL-specific fixes.**
