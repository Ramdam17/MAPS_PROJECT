# Component review — `src/maps/networks/first_order_mlp.py`

**Review sub-phase :** Sprint-08 C.11 (`FirstOrderMLP` + `make_chunked_sigmoid`).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/networks/first_order_mlp.py` (121 lignes, 1 classe + 2 helpers).
**Paper sources :** Table 9 (Blindsight), Table 10 (AGL). §2.2 + §2.3.
**Student sources :**
- `external/paper_reference/blindsight_tmlr.py:136-200` (`FirstOrderNetwork`, Blindsight).
- `external/paper_reference/agl_tmlr.py:134-200` (`FirstOrderNetwork`, AGL).
**Callers du port :**
- `src/maps/experiments/blindsight/trainer.py:184-189` (via `fo_cfg`).
- `src/maps/experiments/agl/trainer.py:147-151` (avec `make_chunked_sigmoid`).
- **SARL n'utilise pas** `FirstOrderMLP` (conv-based `SarlQNetwork`).

**DoD global** : `docs/reviews/first_order_mlp.md` créé, §C.11 complète, 0 code touché.

---

## `FirstOrderMLP` + `make_chunked_sigmoid` (C.11)

Port (121 lignes) :

```python
def make_chunked_sigmoid(chunk_size):
    def _chunked(h):
        out = h.clone()
        for i in range(0, h.size(-1), chunk_size):
            out[..., i:i+chunk_size] = torch.sigmoid(h[..., i:i+chunk_size])
        return out
    return _chunked

class FirstOrderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=40, encoder_dropout=0.1,
                 decoder_activation=_global_sigmoid, weight_init_range=(-1.0, 1.0)):
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(encoder_dropout)
        init.uniform_(self.fc1.weight, *weight_init_range)
        init.uniform_(self.fc2.weight, *weight_init_range)
```

### (a) Architecture conforme student bit-exact

Student Blindsight (`blindsight_tmlr.py:136-200`) + AGL (`agl_tmlr.py:134-200`) :

| Aspect                    | Student Blindsight               | Student AGL                       | Port `FirstOrderMLP`              | Match |
|:--------------------------|:---------------------------------|:----------------------------------|:----------------------------------|:-----:|
| Encoder                   | `fc1 = Linear(100, hidden, bias=False)` | `fc1 = Linear(48, hidden, bias=False)` | `fc1 = Linear(input_dim, hidden_dim, bias=False)` | ✅    |
| Decoder                   | `fc2 = Linear(hidden, 100, bias=False)` | `fc2 = Linear(hidden, 48, bias=False)`  | `fc2 = Linear(hidden_dim, input_dim, bias=False)` | ✅    |
| Hidden activation         | `ReLU` (L155, L177 post-ReLU + Dropout) | idem `ReLU` (L152, L171) | `ReLU` (L94, L101)                | ✅    |
| Encoder dropout           | `0.1` (L160)                     | `0.1` (implicite)                 | `encoder_dropout=0.1` default      | ✅    |
| Decoder activation        | `sigmoid` global (L181)          | `sigmoid` par chunk 6 (L180)       | `decoder_activation=` (paramétrable) | ✅    |
| Bias                      | **False** partout                 | **False** partout                  | **False** partout                  | ✅    |
| Weight init               | `uniform(-1.0, 1.0)` (L173-174)   | idem (L167-168)                    | `uniform(-1.0, 1.0)` default       | ✅    |
| Cascade on h2             | Oui, dans `decoder()` (L184-185)  | idem                               | Oui, via `cascade_update` (L110)  | ✅    |
| Cascade on h1             | **Signature accepte `prev_h1`** mais pas utilisé | idem | Signature accepte mais no-op      | ✅    |

✅ **Port = student bit-exact** sur les 3 domaines + paramétrage `decoder_activation` propre.

### (b) 🚨 Finding C11-F1 — correction C.7/C.10 docstring `cae_loss`

En C.10 j'ai écrit dans le docstring de `components.cae_loss` :

> `h` **must be post-sigmoid** (i.e. in [0, 1]) for the sigmoid-derivative formula `h(1-h)` to be
> mathematically valid.

**C'est faux pour TOUS les domaines** :

- **Blindsight** : encoder `ReLU(fc1(x))` L177 → `h1` post-ReLU, pas sigmoid.
- **AGL** : encoder `ReLU(fc1(x))` L171 → `h1` post-ReLU, pas sigmoid.
- **SARL** : encoder `F.relu(fc_hidden(...))` → post-ReLU.

Donc la formule `h(1-h)` dans `cae_loss` est **paper-faithful quirk dans les 3 domaines**,
pas seulement SARL. Le quirk est **universel** dans la codebase.

**Piste C11-fix-1** (critique) : corriger le docstring `components.cae_loss` — enlever la
section "h must be post-sigmoid" et la remplacer par "h is assumed post-ReLU (student quirk) ;
formula `h(1-h)` is mathematically valid for sigmoid only, but applied on ReLU output for
paper parity in all 3 domains." Cross-ref `docs/reviews/first_order_mlp.md §C.11 (b)`.

### (c) 🚨 Finding C11-F2 — `config/training/blindsight.yaml` diverge de paper Table 9

**Paper Table 9** (verbatim, `paper_tables_extracted.md:121-123`) :
- `Input size = Output size = 100`
- `Hidden size = 60`

**Port config `blindsight.yaml:10`** : `hidden_dim: 100` (**diverge**).
**Port config `agl.yaml:9`** : `hidden_dim: 40` (matche paper Table 10).

**Diagnostic** : le config Blindsight utilise `hidden_dim=100` alors que paper spec `60`.
Déjà tracé dans :
- `deviations.md` : RG-002 H1 (Blindsight metric mismatch + hidden_dim).
- Plan D.25 : "RG-002 fix (Blindsight headline metric)".

**Impact probable** :
- `hidden=100` = identity-ish bottleneck (autoencoder `100 → 100 → 100` avec poids appris sans
  compression dim-wise).
- `hidden=60` = vraie compression `100 → 60 → 100` (paper specs).
- **Peut expliquer partiellement RG-002** (Blindsight z=+0.40 au lieu de +9.01) — le bottleneck
  à 60 force le 1st-order à apprendre une représentation comprimée, ce qui pourrait mieux
  discriminer signal vs noise.

**Piste C11-fix-2** (à faire Phase D.25, pas C.11) : aligner `blindsight.yaml` sur paper
Table 9 (`hidden_dim: 60`). Smoke-test 10 seeds pour comparer z-score avant/après.
**Pas de fix en C.11** — hors scope review.

### (d) Forward signature — `prev_h1` unused

Port forward :
```python
def forward(self, x, prev_h1=None, prev_h2=None, cascade_rate=1.0):
    h1 = self.encoder(x)                        # ← prev_h1 IGNORED
    h2 = self.decoder(h1, prev_h2, cascade_rate) # ← cascade only on h2
    return h1, h2
```

**Student** (verbatim `blindsight_tmlr.py:189`) :
```python
def forward(self, x, prev_h1, prev_h2, cascade_rate):
    # L195-198 implementation
    h1 = self.encoder(x)
    h2 = self.decoder(h1, prev_h2, cascade_rate)
    return h1, h2
```

Student accepte aussi `prev_h1` sans l'utiliser. Port matche.

✅ **Signature paper-faithful** malgré le paramètre inutile — préservé pour l'API.

**Piste C11-fix-3** (cosmétique) : ajouter un warning dans docstring que `prev_h1` est
un no-op. Déjà documenté L73-79. ✅ Rien à faire.

### (e) `make_chunked_sigmoid` vs student AGL

**Student AGL** (`agl_tmlr.py:178-181`) :
```python
h2 = self.fc2(z)
# Apply sigmoid to every 6-unit subpart of each pattern
for i in range(0, h2.shape[1], bits_per_letter):
    h2[:, i:i+bits_per_letter] = self.sigmoid(h2[:, i:i+bits_per_letter])
```

**Port `make_chunked_sigmoid`** (L45-49) :
```python
def _chunked(h):
    out = h.clone()
    for i in range(0, h.size(-1), chunk_size):
        out[..., i:i+chunk_size] = torch.sigmoid(h[..., i:i+chunk_size])
    return out
```

**Différences** :
1. Port `h.clone()` vs student in-place `h2[:, i:i+...] = ...`. Port clone **empêche
   in-place modification** (autograd safer, `h.clone()` est differentiable).
2. Port `...` ellipsis vs student `:` (même sémantique pour 2D).
3. Port préserve gradients correctement (student modifie `h2` in-place, risqué pour autograd si
   `h2` a des usages en amont).

✅ **Gain safety port** ; comportement numérique identique.

**Piste C11-fix-4** (info) : docstring mention "`.clone()` prevents autograd in-place issues
vs student's direct assignment".

### (f) Decoder dropout absent

**Student** : aucune classe n'applique de dropout sur la sortie decoder. Le dropout est **entre
encoder et decoder** (post-ReLU, pre-fc2). Port matche.

**Paper §2.2 / §2.3** : silencieux sur la structure dropout exacte, juste "Dropout: 0.1" dans
Tables 9/10.

✅ **Paper-faithful via student.**

### (g) Relation avec `cae_loss`

Callers :
- Blindsight trainer L328 : `W = self.first_order.fc1.weight` → passé à `cae_loss`.
- AGL trainer L299 : idem.

**Shape de `W`** : `(hidden_dim, input_dim)` — `nn.Linear(in, out).weight.shape = (out, in)`.
Port `FirstOrderMLP.fc1 = Linear(input_dim, hidden_dim)` → `W.shape = (hidden_dim, input_dim)`.

Dans `cae_loss` :
```python
w_sq_rowsum = torch.sum(W_const**2, dim=1, keepdim=True)  # (hidden_dim, 1)
contractive = torch.sum(torch.mm(dh**2, w_sq_rowsum))     # dh: (batch, hidden_dim)
```

**Multiplication `(batch, hidden) × (hidden, 1) = (batch, 1)` → sum = scalar.** OK shape-wise.

✅ **`cae_loss` et `FirstOrderMLP` compatibles shape-wise**, student parity.

### (h) Hidden dim default `40` — piège

Port default `hidden_dim=40` (L85) — c'est la **valeur AGL paper**, pas Blindsight paper (60).
Blindsight caller passe explicitement `hidden_dim=100` (config, actuellement divergent).

**Risque** : si un futur caller oublie de passer `hidden_dim`, il instancie un AGL-sized
Blindsight. Pas un bug, juste un piège docstring.

**Piste C11-fix-5** : clarifier docstring L62-63 que `hidden_dim=40` est l'AGL-default, et que
Blindsight doit passer 60 (paper) ou 100 (config actuel, D-blindsight-hidden-dim). Cross-ref
RG-002.

### Fixes identifiées C.11

| ID        | Fix                                                                                    | Scope                                           | Effort |
|:----------|----------------------------------------------------------------------------------------|-------------------------------------------------|:------:|
| C11-fix-1 🚨 | Correction `components.cae_loss` docstring C.10 — h post-ReLU dans les 3 domaines | `src/maps/components/losses.py:37-66`           | 10 min |
| C11-fix-2 | (Phase D.25, pas C.11) aligner `blindsight.yaml` sur paper Table 9 (hidden 60)         | `config/training/blindsight.yaml:10`            | hors  |
| C11-fix-3 | (skip) docstring prev_h1 no-op — déjà documenté                                        | —                                               | skip   |
| C11-fix-4 | Docstring `make_chunked_sigmoid` : note sur `.clone()` vs student in-place            | `first_order_mlp.py:40-51` docstring            | 3 min  |
| C11-fix-5 | Docstring `FirstOrderMLP.hidden_dim` : clarifier que 40 est AGL-default, Blindsight = 60 (paper) | `first_order_mlp.py:62-63` docstring | 5 min  |

### Cross-reference deviations.md

- **RG-002** (Blindsight hidden_dim divergence) déjà tracé ; la correction reviendra en Phase D.25.
- **Pas de nouvelle entrée** `deviations.md` nécessaire.
- **Cross-ref C.7 errata** : mon docstring `cae_loss` C.10 (h post-sigmoid) est incorrect →
  C11-fix-1 critique.

### Résumé — `FirstOrderMLP`

- ✅ **Architecture bit-exact** avec student Blindsight + AGL sur les 9 checks (a).
- ✅ **Weight init, bias=False, dropout 0.1, ReLU encoder** tous paper-faithful.
- ✅ **`make_chunked_sigmoid`** : gain port (clone vs in-place) sans divergence numérique.
- 🚨 **C11-F1** : `cae_loss` docstring C.10 erroné — h est post-ReLU dans **3 domaines**, pas
  post-sigmoid. Correction C.12 ou hotfix.
- 🚨 **C11-F2** : `blindsight.yaml:10 hidden_dim=100` diverge de paper Table 9 (`60`).
  Tracé RG-002 H1, fix Phase D.25.
- ⚠️ **`hidden_dim=40` default** AGL-coded : piège docstring à clarifier.
- **0 divergence structurelle port-student.**

**C.11 clôturée. 3 fixes pour C.12 batch (C11-fix-1 🚨 critique, C11-fix-4, C11-fix-5), 2 fixes
déjà tracés ailleurs (C11-fix-2 → D.25, C11-fix-3 skip).**
