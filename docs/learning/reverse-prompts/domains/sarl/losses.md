# Reverse-prompt — `src/maps/domains/sarl/losses.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl/losses.py`
**Taille :** ~100 lignes, 1 fonction publique (`cae_loss`).
**Status :** 🚨 **DETTE-2 — Doublon de `core/losses.cae_loss`**.

---

## 1. Quelle aurait été la spec ?

> Écris une fonction Python `cae_loss(W, x, recons_x, h, lam)` qui
> implémente le "CAE loss with Huber reconstruction" du SARL paper.
>
> **Signature trompeuse** : le paper appelle ça `CAE_loss` (Contractive
> AutoEncoder) mais l'usage réel est :
>
> ```python
> loss = cae_loss(W, target, Q_s_a, h1, lam)
> #              ↑  ↑       ↑      ↑   ↑
> #              W  x       recons h   lam
> ```
>
> donc `x = TD target` et `recons_x = Q_s_a` (predicted Q). Le terme
> "Huber recon" mesure `huber(Q_s_a, target)` — c'est la **TD error
> DQN standard**.
>
> **Formule** :
> ```
> loss = F.huber_loss(recons_x, x) + λ · Σ (h(1-h))² · W²
> ```
>
> Le second terme est le Jacobian regularizer Rifai 2011 — même quirk
> que les autres `cae_loss` du projet : `h(1-h)` appliqué sur ReLU
> output (paper-faithful mais mathématiquement faux).

## 2. Pourquoi ce module existe (DETTE-2)

Le port a **deux fonctions `cae_loss` distinctes** :

| Aspect | `core.losses.cae_loss` | `sarl/losses.cae_loss` (this module) |
|--------|------------------------|--------------------------------------|
| Reconstruction term | BCE / MSE (param `recon=`) | **Huber hardcoded** |
| Encoder activation supposed | Sigmoid (formule correcte) | ReLU (formule préservée) |
| Callers | Blindsight, AGL trainers | SARL, SARL+CL trainers |
| Signature | `(W, x, recons_x, h, lam, *, recon="bce_sum")` | `(W, x, recons_x, h, lam)` |

**Pourquoi pas unifier ?**
- Les reconstruction terms sont hétérogènes
- Les activations encoder différentes (en théorie, en pratique tous
  ReLU)
- Unifier demanderait `cae_loss(recon="huber" | "bce_sum" | "mse_*")`

**Quand unifier ?** Post-Phase F. Approche proposée :
```python
def cae_loss(W, x, recons_x, h, lam, *, recon="bce_sum"):
    if recon == "huber":
        recon_term = F.huber_loss(recons_x, x)
    elif recon == "bce_sum":
        recon_term = F.binary_cross_entropy(recons_x, x, reduction="sum")
    # ...
    return recon_term + lam * contractive_term
```

## 3. Contraintes scientifiques

### Huber loss = smooth L1

`F.huber_loss(predicted, target)` avec δ=1.0 (default) :
- `0.5 · (predicted - target)²` si `|err| ≤ 1`
- `|predicted - target| - 0.5` sinon

Plus robuste que MSE aux outliers, plus stable que L1 près de 0.
**Standard DQN loss** (Mnih 2015 utilise Huber, pas MSE pur).

### Pourquoi Huber pour DQN ?

Les Q-values peuvent diverger (rewards aléatoires, exploration
chaotique). Huber clamp les gradients à 1 pour les erreurs > 1,
empêche les updates explosifs.

### Le Jacobian terme

`h(1-h)` × `Σ W²` (Frobenius norm squared du Jacobian sigmoid-style).
Force le 1st-order hidden à être **contractive** (peu sensible aux
perturbations de l'input). C'est ce que Rifai 2011 appelle "contractive
auto-encoder".

**Mathématiquement faux** sur ReLU : `dReLU/dz = 1[z > 0]`, pas
`h(1-h)`. Mais préservé bit-exact pour parity. Peut-être fonctionne
quand même empiriquement.

## 4. Deviations / dettes présentes

### 🚨 DETTE-2 — Doublon

Voir §2. À unifier post-Phase F.

### Quirk `h(1-h)` sur ReLU

Universal dans le code student (Blindsight, AGL, SARL). Préservé.
Voir `core/losses.md` pour discussion détaillée.

### Pas de D-* paper sur ce module

C-* findings du review C.7-C.9 (Sprint-08) sont des nettoyages de
docstring, pas des deviations paper.

## 5. Questions ouvertes

- **Q1 :** Pourquoi Huber dans SARL et BCE dans Blindsight/AGL ? Le
  paper §3 ne le justifie pas explicitement. Mnih 2015 utilise Huber.
  Probable que Juan a copié ça du DQN standard.

- **Q2 :** Le `λ=1e-4` est hardcoded `CAE_LAMBDA` dans `trainer.py`,
  pas exposé en config. Pourquoi ? Peut-être qu'aucune ablation l'a
  varié.

- **Q3 :** Le terme `Σ_j (h(1-h))² · Σ_i W²_ji` n'est pas un vrai
  Jacobien. C'est `Σ_j (∂h_j/∂h_j)² · ||W_j||²` qui est juste une
  scale arbitraire. Le vrai contractive serait
  `Σ_ji (∂h_j/∂x_i)² = Σ_ji (h_j(1-h_j))² · W²_ji`. C'est ce que le
  code calcule en réalité via `torch.mm(dh², W².T)`. **À vérifier
  formellement.**

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Math identique (Huber + Jacobian term)
- Signature `(W, x, recons_x, h, lam)` positional (parity callers)
- Pas de `recon=` param ici (paper-faithful)

### À unifier (post-Phase F)
- Merger dans `core/losses.cae_loss` avec `recon="huber"`
- Supprimer ce fichier ou le rendre un wrapper :
  ```python
  from maps.core.losses import cae_loss as _cae_loss
  def cae_loss(W, x, recons_x, h, lam):
      return _cae_loss(W, x, recons_x, h, lam, recon="huber")
  ```

### Tests à écrire
- Parity bit-exact vs `external/paper_reference/sarl/maps.py`
- Vérifier que Huber est utilisé (pas MSE), avec δ=1.0 default
- Vérifier que λ=0 → loss = huber only

### Connexion
- Appelé par `trainer.sarl_update_step`
- Appelé par `domains/sarl_cl/trainer` (DETTE-2 cross-domain)
- Sera unifié post-Phase F dans `core/losses.py`
