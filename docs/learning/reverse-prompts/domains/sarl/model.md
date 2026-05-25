# Reverse-prompt — `src/maps/domains/sarl/model.py` (v2)

**Module path actuel (sur main) :** `src/maps/experiments/sarl/model.py`
**Taille :** ~220 lignes.
**Status :** **v2 secondary variant** (Sprint-04b port). **v1 est
canonical** depuis Sprint-09 (`model_v1.py`).
**Paper :** §2.1, §3.

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `model.py` (v2) qui porte les networks SARL
> depuis `external/paper_reference/sarl/maps.py` (= `maps_v2.py` de
> Juan). **Note : ce port a été fait à tort** lors de Sprint-04b — le
> shell `SARL_Training_Standard.sh` utilise en réalité `maps_v1.py`
> (Sprint-09 D-sarl-wrong-variant resolution).
>
> Néanmoins on garde v2 pour ablations et historique Phase F.4 archived
> runs.
>
> **2 classes** :
>
> **1. `SarlQNetwork`** (first-order Q-network) :
> - `Conv2d(in_channels, 16, kernel=3, stride=1)` → ReLU
> - `Linear(NUM_LINEAR_UNITS=1024, 128)` (fc_hidden) → ReLU
> - **Tied-weight reconstruction** : `F.linear(hidden, fc_hidden.weight.t(),
>   b_recon)` → ReLU. Le decoder réutilise les poids encoder (eq.12 paper).
> - **Cascade sur Hidden (128-d)**, PAS sur Output
> - Q-head : `Linear(128, num_actions)`
> - Forward returns `(q_values, hidden, comparison, hidden_copy)` où
>   `comparison = flat_input - reconstruction` (eq.1 inline)
>
> **`b_recon` ajouté Sprint-08 D.7** : `nn.Parameter(torch.zeros(1024))`.
> Zero-init préserve forward parity initiale, puis apprend via
> gradients. Resolution D-sarl-recon-bias (paper eq.12).
>
> **2. `SarlSecondOrderNetwork`** (second-order, v2) :
> - **Pas de `comparison_layer`** — directe dropout sur comparison_matrix
> - `Dropout(p=0.1)` → cascade_update → `Linear(1024, 2)` (wager)
> - Returns **raw logits** (B, 2) — pas sigmoid/softmax. Downstream
>   `binary_cross_entropy_with_logits`.
> - Weight init uniform(0, 0.1)

## 2. Contraintes scientifiques

### Architecture vs v1

| Aspect | v2 (ce module) | v1 (`model_v1.py`) |
|--------|----------------|---------------------|
| Decoder | Tied-weight `F.linear(W.t())` | Dedicated `Linear(128, 1024)` |
| Cascade target | Hidden (128-d) | Output (1024-d) |
| Q-head input | Hidden (128-d) | Output (1024-d) |
| `b_recon` | Added separate param | Built-in via decoder bias |
| comparison_layer (2nd-order) | Absent | Active `Linear(1024, 1024)` |
| Training budget (paper) | 500k frames, α=0.45 | 2M frames, α=0.25 |
| Status canonical | ❌ wrong variant Sprint-04b | ✅ paper-canonical Sprint-09 |

### Paper §2.1 eq.12 (tied-weight)

`Ŷ^{(1)} = ReLU(fc_hidden.weight.T · hidden + b_recon)`. Le student
v2 omettait `b_recon` (bias=False implicit). On l'a ajouté Sprint-08
D.7 zero-init pour forward parity initiale.

### Cascade no-op sur Q-network

`SarlQNetwork.forward` **n'a pas de dropout** dans le path Hidden
calcul. Donc cascade(Hidden, prev_h2, α) sur 50 iters avec mêmes
inputs → Hidden constant → cascade = no-op math (D-sarl-cascade-noop,
cf. `core/cascade.md`).

Active path 2nd-order (a dropout) — cascade active.

## 3. Contraintes d'ingénierie

### Layer construction order matters

PyTorch draw les weights from default generator sequentially. Si on
swap l'ordre de `nn.Conv2d` / `nn.Linear` constructors, les seeds
identiques produisent des weights différents. **Donc ordre des
constructors locked**.

### `b_recon = nn.Parameter(torch.zeros(1024))`

`torch.zeros` ne consomme PAS de RNG — donc ajouter ce param ne
shift PAS la sequence des draws pour les autres layers. Tier 1/3
parity tests restent bit-exact à init.

### `NUM_LINEAR_UNITS = 1024`

Computed via `_size_linear_unit(10) * _size_linear_unit(10) * 16`
= `8 * 8 * 16` = 1024. MinAtar grids 10×10, conv 3×3 stride 1 →
8×8 output, 16 filters.

### Forward returns 4-tuple

`(q_values, hidden, comparison, hidden_copy)` — 4ème slot est juste
une copie de hidden, redondant mais matches paper signature.

### Asymmetric : encoder pas cascade

Cascade sur `hidden` (post-MLP, 128-d), pas sur `conv_out`. Donc la
conv stage est recalculée à l'identique chaque iteration. Inefficiency
mais paper-faithful.

## 4. Deviations / dettes présentes

### 🚨 D-sarl-wrong-variant (Sprint 09 resolved)

v2 = wrong variant. v1 (`model_v1.py`) est canonical. Voir
`model_v1.md`.

### ✅ D-sarl-recon-bias (D.7 resolved)

`b_recon` ajouté zero-init. Forward parity at init OK.

### ⚠️ D-sarl-cascade-noop (D.4 documented)

Cascade no-op sur path déterministe (pas de dropout). Logged warning,
non-fix per paper-faithful policy.

### DETTE-1 — Duplication avec `core/second_order.py`

`SarlSecondOrderNetwork` (ce module) vs `SecondOrderNetwork` (core).
Différences :
- Pas de `ComparatorMatrix` (comparison inline dans Q-network)
- Dropout 0.1 vs 0.5
- Wager dims `(1024, 2)` hardcoded vs param
- Raw logits vs sigmoid

À unifier post-Phase F via `SecondOrderCore(...)` + wrappers.

## 5. Questions ouvertes

- **Q1 :** Pourquoi tied-weight v2 vs dedicated v1 ? Tied = moins de
  params (1024×128 partagé), souvent vu en autoencoder. Mais le paper
  spec dedicated v1. **Le choix tied (v2) est défaut empirique non
  paper-faithful**.

- **Q2 :** Cascade sur Hidden vs Output : v2 vs v1. Hidden cascade
  est plus "compact" (128-d sequential ops) mais sémantiquement
  différent (cascade sur la représentation comprimée vs la
  reconstruction).

- **Q3 :** Le `b_recon` a une signification physique (offset
  reconstruction) mais zero-init le rend "découvert" via training.
  Quelle valeur final il prend en pratique ? Test : log b_recon norm
  pendant training.

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder (pour ablations / archived Phase F.4)
- Tied-weight decoder
- `b_recon` zero-init
- Cascade sur Hidden
- Pas de comparison_layer dans 2nd-order
- Raw logits wager output

### À unifier post-Phase F
- Avec `core/second_order.SecondOrderNetwork` via DETTE-1 resolution

### Tests à écrire
- Parity tier 1/3 bit-exact vs `external/paper_reference/sarl_maps.py`
  (= maps_v2.py)
- b_recon shape (1024,), zero-init
- Forward returns 4-tuple correct shapes
- Cascade no-op sur Q-network (50 iters = 1 iter)

### Connexion
- Utilisé par `training_loop.py` quand `model_variant="v2"`
- Tests parity tier 1/3 dans `tests/parity/sarl/`
