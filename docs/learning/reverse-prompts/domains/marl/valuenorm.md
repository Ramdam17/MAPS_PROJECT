# Reverse-prompt — `src/maps/domains/marl/valuenorm.py`

**Module path actuel :** `src/maps/experiments/marl/valuenorm.py`
**Taille :** ~110 lignes.
**Référence :** Yu et al. 2022 ("Surprising Effectiveness of PPO in MARL").

---

## 1. Quelle aurait été la spec ?

> Écris une class `ValueNorm` pour stabiliser PPO value loss via
> running-mean / running-variance normalisation. Port verbatim student
> `onpolicy/utils/valuenorm.py`.
>
> Active via `cfg.ppo.use_valuenorm = True` (default port).
>
> **API** :
> - `__init__(input_shape, norm_axes=1, beta=0.99999,
>   per_element_update=False, epsilon=1e-5)`
> - `update(input_vector)` : update running mean / mean² via EMA
> - `normalize(input_vector)` : `(x - mean) / sqrt(var)`
> - `denormalize(input_vector)` : `x · sqrt(var) + mean`
> - `running_mean_var()` : return debiased (mean, var)
>
> **Internal state** (`nn.Parameter` with `requires_grad=False`) :
> - `running_mean` (input_shape)
> - `running_mean_sq` (input_shape)
> - `debiasing_term` (scalar, EMA of "weight" applied)
>
> **EMA recurrence** :
> ```
> running_mean = β · running_mean + (1-β) · batch_mean
> running_mean_sq = β · running_mean_sq + (1-β) · batch_mean_sq
> debiasing_term = β · debiasing_term + (1-β) · 1.0
> ```
>
> **Debiased values** (Kingma init bias correction) :
> ```
> debiased_mean = running_mean / clamp(debiasing_term, ε)
> debiased_var = debiased_mean_sq / clamp(debiasing_term, ε) - debiased_mean²
> debiased_var = clamp(debiased_var, 1e-2)  # avoid sqrt(0)
> ```
>
> **`per_element_update`** : if True, weight = β^batch_size (per-element
> update). Else weight = β (per-batch update). Default False.

## 2. Contraintes scientifiques

### Pourquoi normaliser les values ?

MAPPO/PPO value loss `MSE(V, returns)`. Si returns varient sur des
scales 0-1000 (e.g. cumulative reward), la loss domine par les
samples avec grosses returns. Adam s'adapte mais lentement.

Normalize les values + denormalize pour predict → optimization plus
stable.

### `beta = 0.99999` very high

Permet long histoire (~ 100k samples avant que EMA "oublie"). Plus
lent que `beta=0.999` standard mais plus stable estimate sur RL où
les distributions de returns changent.

### Debiased EMA (Kingma + Adam)

Naive EMA :
```
ema_t = β · ema_{t-1} + (1-β) · x_t
```
À t=0, ema_0 = (1-β)·x_0 = très petit (β=0.99999 → 1e-5). Biaisé.

Debiased :
```
ema_t / (1 - β^t) ≈ unbiased mean
```

Notre `debiasing_term` accumule `(1-β)` chaque step → converge à 1.
Division par lui démarre un step 1.

## 3. Contraintes d'ingénierie

### `nn.Parameter` avec `requires_grad=False`

Permet de mettre running stats dans `state_dict()` pour
serialization. Sans gradient flow.

### `denormalize → numpy`

Pour API compat avec student qui retournait numpy. Notre code pourrait
être plus type-clean (tensor only) mais break parity caller.

### `epsilon = 1e-5` denom safety

Évite `0/0` quand `debiasing_term` est 0 (t=0 avant first update).

### `clamp(var, 1e-2)`

Évite `sqrt(0)`. Si la variance estimée est < 1e-2, on clamp à 1e-2.
Petit biais introduit mais évite NaN.

## 4. Deviations / dettes présentes

Pas de déviation paper sur ce module.

**Pas de DETTE active.**

## 5. Questions ouvertes

- **Q1 :** `beta=0.99999` very high — paper-confirmed ? Student
  default. Plus reactive ?

- **Q2 :** `clamp(var, 1e-2)` clamp threshold raisonnable ? Si les
  values vrais sont < 0.1 stddev, on biais.

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Running mean/mean²/debiasing_term as nn.Parameter requires_grad=False
- Debiased EMA formula
- `beta=0.99999` default
- numpy return on denormalize (parity)

### À améliorer
- **Type hints `Tensor | np.ndarray`** explicit
- **Optional tensor-only mode** for cleaner API

### Tests à écrire
- `update + normalize + denormalize` round-trip identity (within
  precision)
- Debias correct at t=1, t=10, t=10000
- `state_dict` includes running stats
- `clamp(var, 1e-2)` fires when needed

### Connexion
- Utilisé par : `trainer.MAPPOTrainer.cal_value_loss`
- Utilisé par : `data.RolloutBuffer.compute_returns` (denormalize
  value_preds avant GAE)
