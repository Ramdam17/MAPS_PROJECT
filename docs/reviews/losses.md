# Component review — `src/maps/components/losses.py` + `src/maps/experiments/sarl/losses.py`

**Review sub-phases :** Sprint-08 C.7 (`cae_loss` + D-002 deep dive), C.8 (`wagering_bce_loss`,
`distillation_loss`), C.9 (`weight_regularization` + cross-ref callers).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-19.
**Files under review :**
- `src/maps/components/losses.py` (~215 lignes, 4 fonctions).
- `src/maps/experiments/sarl/losses.py` (~91 lignes, 1 fonction — duplicat `cae_loss`).
**Paper sources :** eq. 4 (contrastive loss), eq. 5 (BCE wagering), eq. 17 (CL weighted), Table 11.
**Student sources :**
- `external/paper_reference/blindsight_tmlr.py:87-129` (`CAE_loss` avec `mse_loss = BCELoss(size_average=False)`).
- `external/paper_reference/agl_tmlr.py:85-127` (même template).
- `external/paper_reference/sarl_maps.py:330-379` (CAE_loss avec Huber reconstruction).
**Callers du port (cae_loss) :**
- `src/maps/experiments/blindsight/trainer.py:329`.
- `src/maps/experiments/agl/trainer.py:300`.
- `src/maps/experiments/sarl/trainer.py:188` (version Huber).
- `src/maps/experiments/sarl_cl/trainer.py:283, 302` (version Huber).

**DoD global** : 1 fichier nouveau (`losses.md`), 3 sections (C.7/C.8/C.9), D-002 tranché.

---

## `cae_loss` (C.7)

Deux implémentations coexistent dans le port :

1. **`src/maps/components/losses.cae_loss`** (Blindsight + AGL) — `recon="bce_sum" | "mse_mean" | "mse_sum"`.
2. **`src/maps/experiments/sarl/losses.cae_loss`** (SARL + SARL+CL) — `F.huber_loss` hardcodé.

Les deux portent le même nom mais sont **fonctionnellement distinctes** (reconstruction term
différent). Duplication tracable → DETTE-2 proposée.

### (a) 🚨 D-002 — paper "contrastive" (SimCLR) vs student/port "CAE" (Rifai) : divergence réelle

**Paper eq. 4** (verbatim, `docs/reproduction/paper_equations_extracted.md:71-88`) :

$$
\mathcal{L}_{\mathrm{contrastive}} = \ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k\ne i]} \exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}
$$

- **Référence** : Chen et al. 2020 SimCLR, NT-Xent loss.
- **Variables** : `z_i, z_j` paires positives (augmentations du même sample), `2N` pairs dans le
  batch, `τ` température, `sim(·,·)` cosine similarity typique.

**Student / Port** : Contractive AutoEncoder loss (Rifai et al. 2011) :

$$
\mathcal{L}_{\mathrm{CAE}} = \underbrace{\mathrm{recon}(x, \hat{x})}_{\text{BCE ou Huber}} + \lambda \cdot \underbrace{\|J_h(x)\|^2_F}_{\text{contractive penalty}}
$$

où `||J_h(x)||²_F = Σ_j (h_j(1-h_j))² · Σ_i W_ji²` (forme analytique sous sigmoid encoder).

**Les 2 losses sont mathématiquement distinctes** :

| Aspect        | SimCLR (paper eq.4)                                 | CAE (student/port)                             |
|:--------------|:----------------------------------------------------|:-----------------------------------------------|
| Objectif      | Attirer paires positives / repousser négatives     | Reconstruire + pénaliser Jacobien encoder      |
| Paires        | `(z_i, z_j)` augmentations                          | `(x, x̂)` input vs reconstruction              |
| Gradient via  | Normalized embeddings + softmax                    | Reconstruction error + sigmoid-derivative      |
| Batch-dep     | Oui (négatifs du batch)                            | Non (per-sample)                                |
| Hyperparams   | τ (température)                                     | λ (poids jacobien)                             |

**Conclusion** : D-002 est une **divergence paper-vs-student réelle** — pas une collision de
noms. Le student docstring (`external/paper_reference/blindsight_tmlr.py:98-101` verbatim) écrit
*"Contrastive loss plays a crucial role in maintaining the similarity and correlation of latent
representations across different modalities"* — **phraséologie SimCLR/contrastive learning**
appliquée à une implémentation CAE. Confusion terminologique du student, mais **l'implémentation
est CAE**.

**Implication** : le student code **ne peut pas** produire les chiffres paper Tables 5/6/7 sur ce
point (déjà documenté dans Phase B.7-B.11 audit). Les runs originaux paper ont **soit** (a) utilisé
une version différente du code avec SimCLR implémenté, **soit** (b) les chiffres paper sont faux
quant à la loss déclarée.

### (b) Policy trancher D-002 — verrou 2026-04-19 `paper = source of truth`

**Option 1 — Implémenter SimCLR (paper-faithful)** :
- Coût : nouveau module `maps.components.losses.contrastive_simclr` + impl augmentation
  pipeline (SimCLR requiert paires augmentées, non-trivial pour Blindsight patterns binaires et
  AGL letter chunks).
- Risque : **aucune parité** avec student possible (losses différentes) → Tier-1 tests régressent.
- Effort : ~1-2 jours de travail (impl + tests + calibration τ).
- **Bénéfice** : paper-faithful reproduction.

**Option 2 — Rester sur CAE (student parity)** :
- Coût : 0 (déjà fait).
- Risque : reproduction **non paper-faithful** ; Tables 5/6/7 impossibles à atteindre.
- Transparence : déjà documenté D-002.

**Recommandation C.7** : **ne pas trancher en review**. Créer `Phase D.17` (ou équivalent) dédiée
à la décision SimCLR vs CAE avec Rémy + revue de l'évidence :
1. Avons-nous des traces dans paper appendix que les runs utilisaient CAE et pas SimCLR ?
2. Les auteurs originaux (Juan, Zahra) sont injoignables → pas de source primaire.
3. **Fallback policy** : si SimCLR impl ne reproduit pas Tables 5/6/7 non plus, report honnête en
   Phase G `"neither CAE nor SimCLR reproduces paper targets"`.

**Pour C.7 review** : on reste **descriptif**, on documente les 2 options, on renvoie la décision
à Rémy + Phase D.

### (c) Port vs student — parité bit-exact du chemin CAE (Blindsight/AGL)

Port `components.losses.cae_loss` vs student `blindsight_tmlr.py:91-129` :

| Step                         | Student L120-129                                      | Port L68-87                                           | Match |
|------------------------------|-------------------------------------------------------|-------------------------------------------------------|:-----:|
| Reconstruction term          | `mse = mse_loss(recons_x, x)` (où `mse_loss = BCELoss(size_average=False)`) | `F.binary_cross_entropy(recons_x, x, reduction="sum")` | ✅ (API moderne) |
| Hidden sigmoid-derivative    | `dh = h * (1 - h)`                                    | `dh = h * (1 - h)`                                    | ✅    |
| Weight Jacobian term         | `w_sum = torch.sum(W**2, dim=1, keepdim=True)`        | `w_sq_rowsum = torch.sum(W_const**2, dim=1, keepdim=True)` | ✅    |
| Contractive sum              | `contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)` | `contractive = torch.sum(torch.mm(dh**2, w_sq_rowsum))` | ✅    |
| Lambda scaling + return      | `return mse + contractive_loss.mul_(lam)`             | `return recon_term + lam * contractive`              | ✅ (même valeur, impl différente) |

✅ **Port = student bit-exact** sur le chemin `recon="bce_sum"`. Les variantes `mse_mean` / `mse_sum`
sont des extensions port (non-student), **safe** tant que caller passe `recon="bce_sum"`.

### (d) `W.detach()` — subtilité matchant le student

Port L83 : `W_const = W.detach()`.
Student L602 : `W = first_order_network.state_dict()['fc1.weight']`.

**PyTorch `state_dict()` default (`keep_vars=False`)** : appelle `detach()` sur chaque paramètre
avant de les mettre dans l'OrderedDict. → Student W est **déjà détaché**. Port `W.detach()` **matche
ce comportement exactement**.

**Conséquence** : le terme contractif ne propage **pas** de gradient vers W directement ; seulement
vers `h` (et à travers le sigmoid-derivative vers les poids du 1st-order par backprop de `h`).

✅ **`W.detach()` paper-faithful via matching student state_dict behavior.**

⚠️ **Note incidente** : le docstring de `sarl.losses.cae_loss` lignes 31-36 dit l'**inverse** :

> ``W`` is taken live from ``policy_net.state_dict()['fc_hidden.weight']``. state_dict returns
> the actual parameter tensor (no detach), so gradients from the Jacobian term DO propagate back
> to ``fc_hidden.weight``.

**C'est factuellement faux pour state_dict default** (`keep_vars=False` détache toujours). Soit le
student SARL utilise `keep_vars=True` (improbable), soit le docstring port SARL est erroné. À
vérifier en C.8 via `external/paper_reference/sarl_maps.py`. Flagge **C7-fix-4** pour update du
docstring SARL.

### (e) Assumptions non-documentées

Le port ne valide pas :
1. **`recons_x ∈ [0, 1]`** requis par `binary_cross_entropy`. Fail bruyant si decoder sort des
   logits. Blindsight decoder (`first_order_mlp.py`) utilise sigmoid → OK. AGL decoder aussi (via
   chunked sigmoid D-004) → OK. Mais fragile à un refacto.
2. **`h ∈ [0, 1]`** requis pour que `h(1-h)` représente la dérivée sigmoid. Blindsight/AGL student
   utilise sigmoid → OK. **SARL utilise ReLU** (student `sarl_maps.py`) → formule `h(1-h)` est
   **mathématiquement fausse** pour ReLU (la dérivée de ReLU est `1[h > 0]`, pas `h(1-h)`).
   Student le fait quand même → port le préserve pour parity. Documenté dans
   `sarl/losses.py:37-40`. ⚠️ **Quirk à garder, pas un bug.**
3. **`h.shape == (B, N_hidden)`** requis pour `torch.mm(dh**2, w_sq_rowsum)` — pas vérifié.

**Piste C7-fix-1** : ajouter une mention dans le docstring `components.losses.cae_loss` que
`recons_x` doit être post-sigmoid ET `h` doit être post-sigmoid (dérivée formulation). Pas de
runtime assertion — coût/bénéfice marginal.

**Piste C7-fix-2** : docstring explicite sur l'asymétrie SARL (ReLU hidden, formule quand même
`h(1-h)`).

### (f) Duplication `components.losses.cae_loss` vs `experiments.sarl.losses.cae_loss` — DETTE-2

Le port a **2 fonctions `cae_loss`** différentes :

| Aspect             | `components.losses.cae_loss`        | `sarl.losses.cae_loss`                 |
|:-------------------|:------------------------------------|:---------------------------------------|
| Reconstruction     | BCE (default) / MSE mean / MSE sum | Huber (hardcoded)                      |
| Hidden activation  | Sigmoid (formule correcte)         | ReLU (formule préservée malgré fausseté) |
| Callers            | Blindsight + AGL trainers           | SARL + SARL+CL trainers                |
| Signature          | `(W, x, recons_x, h, lam, *, recon="bce_sum")` | `(W, x, recons_x, h, lam)` |

**Justification de la duplication** : les 3 reconstruction terms (BCE / Huber / MSE) sont
hétérogènes, et le type d'activation du hidden diffère. Unifier demanderait :
- Soit un `recon="huber"` dans `components.losses.cae_loss` (trivial, 3 lignes).
- Soit factorisation en `cae_generic(W, x, recons_x, h, lam, recon_fn, hidden_activation)`.

**Recommandation** : **consolider après Phase F** (reproduction validée). DETTE-2 candidate.

**Piste C7-fix-3** : ajouter DETTE-2 dans `deviations.md` pour tracking. Pas d'unification en C.7.

### Fixes identifiées C.7 (pour C.10 batch)

| ID        | Fix                                                                                    | Scope                                           | Effort |
|:----------|----------------------------------------------------------------------------------------|-------------------------------------------------|:------:|
| C7-fix-1  | Docstring `components.cae_loss` : noter que `recons_x` + `h` doivent être post-sigmoid | `losses.py:37-66` docstring                    | 5 min  |
| C7-fix-2  | Docstring : asymétrie SARL (h ReLU, formule h(1-h) préservée pour parity)              | `sarl/losses.py:56-79` docstring (déjà OK)      | 2 min  |
| C7-fix-3  | Ajouter DETTE-2 (duplication cae_loss) dans `deviations.md`                            | `docs/reproduction/deviations.md` DETTE section | 5 min  |
| C7-fix-4  | Correction docstring `sarl.losses.cae_loss` lines 31-36 : state_dict default détache (détails : clarifier que grad ne flow PAS directement vers W) | `sarl/losses.py:31-36` | 5 min  |
| C7-fix-5  | Créer Phase D.17 (ou équivalent) : décision SimCLR vs CAE pour D-002                   | `docs/plans/plan-20260419...` ajout sub-phase   | 10 min |

### Cross-reference deviations.md

- **D-002 déjà ouvert** : description actuelle (après C.6) mentionne cascade interaction mais pas
  la policy trancher. Mise à jour C.10 : ajouter "résolution Phase D.17 (ou C.7-delegate)".
- **DETTE-2 nouveau** : duplication `cae_loss` Blindsight/AGL vs SARL.

### Résumé — `cae_loss`

- 🚨 **D-002 divergence paper-vs-student réelle** (SimCLR vs CAE, mathématiquement distinctes).
  Decision déléguée à Phase D.17.
- ✅ **Port = student bit-exact** sur le chemin CAE (Blindsight/AGL `bce_sum`).
- ✅ **`W.detach()`** matche le `state_dict()` default du student.
- ⚠️ **Assumptions silencieuses** : `recons_x ∈ [0,1]`, `h ∈ [0,1]` pour Blindsight/AGL,
  `h(1-h)` sur ReLU pour SARL (quirk paper-faithful).
- ⚠️ **Duplication** `components.cae_loss` + `sarl.cae_loss` → DETTE-2, à consolider post Phase F.
- ⚠️ **Docstring SARL erronée** sur state_dict behavior → C7-fix-4.
- **1 divergence structurelle (D-002), 1 dette (DETTE-2), 3 fixes docstring.**

**C.7 clôturée. 5 fixes + 1 update deviations.md + 1 nouvelle sub-phase D.17 à proposer pour C.10.**

---

## `wagering_bce_loss` + `distillation_loss` (C.8)

### `wagering_bce_loss` (L90-123)

Port ~34 lignes :

```python
def wagering_bce_loss(wager, target, reduction="mean", pos_weight=None):
    return F.binary_cross_entropy(
        wager, target.to(wager.dtype), weight=pos_weight, reduction=reduction,
    )
```

#### (a) Conformité paper eq.5 + student

**Paper eq.5** (verbatim, `paper_equations_extracted.md:101-103`) :

$$\mathcal{L}_{\mathrm{BCE}} = -[y \cdot \log(\sigma(\mathrm{logits})) + (1-y) \cdot \log(1 - \sigma(\mathrm{logits}))]$$

Paper attend donc `binary_cross_entropy_with_logits` (sigmoid INSIDE la loss).

**Student** (`blindsight_tmlr.py:430, 573`) :
```python
criterion_2 = nn.BCELoss(size_average=False)
...
loss_2 = criterion_2(output_second_order.squeeze(), order_2_tensor[:, 0])
```

où `output_second_order = self.sigmoid(self.wager(...))` L250 — sigmoid déjà appliqué AVANT la
loss. Donc student utilise **BCE-on-probs**, pas BCE-with-logits.

**Port** : `F.binary_cross_entropy(wager, target, ...)` — même approche que student (prend des
probs post-sigmoid).

✅ **Port = student**, mais **≠ paper eq.5** (paper écrit σ(logits) dans la formule, student et port
appliquent σ en amont).

**Note** : mathématiquement équivalent (BCE(σ(x), y) = BCE-with-logits(x, y)), mais
`binary_cross_entropy_with_logits` est **numériquement plus stable** (log-sum-exp trick). Pour
training from scratch, la version logit-based est préférable. C'est **documenté dans le port
docstring L99-101** comme piste "parity only, prefer logit variant for new training".

#### (b) ⚠️ Compatibilité avec `WageringHead(n_wager_units=2)` post-C.6

Post-C.6, `WageringHead(n_wager_units=2)` retourne **raw logits**. Si un caller appelle
`wagering_bce_loss(wager_logits, target)`, la BCE sur logits **bruts** (non-clampés à [0,1]) va
fail bruyamment :

```python
F.binary_cross_entropy(logits_raw_outside_[0,1], target, ...)
# RuntimeError: all elements of input should be between 0 and 1
```

**Port L98** documente *"Takes probabilities (the sigmoid output of `WageringHead(n_wager_units=1)`)"*
— explicite sur l'exigence probs. Mais **aucune guard runtime**.

**Callers actuels** :
- `blindsight/trainer.py:311` : `wagering_bce_loss(wager.squeeze(-1), ...)` où `wager =
  self.second_order(...)[0]` avec `n_wager_units=1` → sigmoid → prob. OK.
- `agl/trainer.py:282` : idem. OK.

Aucun caller actuel n'utilise `n_wager_units=2`, donc pas de bug prod. Mais piège à documenter.

**Piste C8-fix-1** : ajouter dans le docstring un paragraphe *"Incompatible with
`WageringHead(n_wager_units=2)` raw logits post-C.6. Use `F.binary_cross_entropy_with_logits`
instead for the 2-unit paper-faithful path."*

#### (c) 🚨 `pos_weight` mal-nommé — bug API

**Port L121** : `F.binary_cross_entropy(wager, target, weight=pos_weight, reduction=...)`.

Le paramètre `pos_weight` est ici *passé comme* `weight=`. Mais :
- `F.binary_cross_entropy` prend `weight` = poids **par sample** (shape batch).
- `F.binary_cross_entropy_with_logits` prend `pos_weight` = poids **par classe positive** (scalaire).

Le port utilise le **nom `pos_weight`** (qui suggère class imbalance, API
`binary_cross_entropy_with_logits`) mais le **passe à `weight=`** de `binary_cross_entropy` qui
attend per-sample. **Confusion API : le nom suggère une sémantique différente de l'impl.**

Si un caller passe `pos_weight=torch.tensor(5.0)` en pensant "poids sur la classe positive", il
obtient un broadcast sur tous les éléments → tous les samples sont pondérés ×5, pas seulement les
positifs. **Silent bug.**

**Callers actuels** : **aucun** n'utilise `pos_weight` (tous None). Pas de bug prod.

**Piste C8-fix-2** : **renommer** le paramètre `pos_weight` → `weight` pour matcher
l'API `F.binary_cross_entropy`. Ou **supprimer** le paramètre tant que personne ne s'en sert
(YAGNI — minimalisme). Vérifier absence de callers externes avant retrait.

#### (d) Signature et types

- `target.to(wager.dtype)` : cast défensif, OK.
- `reduction="mean"` default : student utilise `size_average=False` (= `reduction="sum"` en
  modern API). **Default port diffère du student sur reduction.** Callers passent
  explicitement `reduction="sum"` (blindsight/AGL) donc pas de bug prod — mais le default
  est trompeur.

**Piste C8-fix-3** (debate) : changer default `reduction="mean"` → `reduction="sum"` pour matcher
student. **Mais** `"mean"` est le default PyTorch standard, et les 2 callers passent explicitement
`"sum"`. Laisser `"mean"` et documenter.

#### Résumé `wagering_bce_loss`

- ✅ **Parity student** : BCE-on-probs, même comportement que `nn.BCELoss(size_average=False)`.
- ✅ **Parity paper eq.5** : mathématiquement équivalent (σ pré-appliqué vs σ INSIDE la loss).
- ⚠️ **Incompatible avec `n_wager_units=2`** raw logits — documenter (C8-fix-1).
- 🚨 **Paramètre `pos_weight` mal nommé** — passé à `weight=`, API confuse (C8-fix-2).
- ⚠️ **Default `reduction="mean"`** diffère de l'usage student `"sum"` (pas un bug, callers
  passent explicitement).

### `distillation_loss` (L126-170)

Port ~45 lignes :

```python
def distillation_loss(student_logits, teacher_logits, hard_labels=None, alpha=0.5, temperature=2.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = torch.sum(
        soft_targets * (soft_targets.clamp_min(1e-12).log() - log_probs), dim=-1
    ).mean()
    if hard_labels is None:
        return alpha * soft_loss
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

#### (a) Conformité student SARL+CL

**Student** (`sarl_cl_maps.py:360-405`) :

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        self.T = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, hard_labels=None, alpha=0.5):
        soft_targets = f.softmax(teacher_outputs / self.T, dim=-1)
        log_probs = f.log_softmax(student_outputs / self.T, dim=-1)
        # Comment: "Calculate KL divergence loss (scaled by T²)"  ← ⚠️ COMMENT LIE
        soft_loss = torch.sum(soft_targets * (soft_targets.log() - log_probs), dim=-1).mean()
        hard_loss = 0
        if hard_labels is not None:
            hard_loss = self.criterion(student_outputs, hard_labels)
        return alpha * soft_loss + (1 - alpha) * hard_loss
```

**Mapping port ↔ student** :

| Step              | Student                                  | Port                                      | Match |
|-------------------|------------------------------------------|-------------------------------------------|:-----:|
| Soft targets      | `softmax(teacher / T)`                   | `softmax(teacher / T)`                    | ✅    |
| Log probs student | `log_softmax(student / T)`               | `log_softmax(student / T)`                | ✅    |
| Soft loss formula | `Σ soft * (soft.log() - log_probs)).mean()` | `Σ soft * (soft.clamp_min(1e-12).log() - log_probs)).mean()` | ✅ (port add clamp, numerical safety) |
| Hard loss         | `nn.CrossEntropyLoss()(student, labels)` | `F.cross_entropy(student, labels)`        | ✅    |
| Combination       | `alpha * soft + (1-alpha) * hard`         | idem                                      | ✅    |

✅ **Port = student + clamp stabilization**. La `clamp_min(1e-12)` est un **gain port** (pas une
régression) — évite `-inf` si `soft_targets` contient des zéros après softmax (impossible en
float32 en pratique, mais garde-fou).

#### (b) ⚠️ D-003 confirmé — et le student ment dans son commentaire

**Paper implicit / Hinton 2015** : `soft_loss *= T²` pour préserver les magnitudes de gradient à
travers les températures.

**Student commentaire L394** : *"Calculate KL divergence loss (scaled by T²)"* — **MENT** : le
code L395 ne scale **PAS** par T². Le commentaire est faux.

**Port** : ne scale pas non plus (matches student impl). D-003 existant confirmé : le port est
fidèle au student, qui diverge du paper/Hinton.

**Impact** : gradient magnitudes ~1/T² plus petites que Hinton-faithful. Dans SARL+CL, `T=2.0`
→ gradient 4× plus petit qu'attendu. **Peut affecter convergence** mais pas documenté comme
problème dans runs student → probablement absorbé par `alpha` weighting ou learning rate.

**Pas de fix en C.8** : D-003 existant, status quo OK (matches student).

**Piste C8-fix-4** : vérifier que le **docstring port** L159-161 documente correctement cette
divergence :

```python
# KL(teacher || student) as written in the reference code, not scaled by T².
# Kept as-is for parity with SARL+CL; if we later want Hinton-style
# gradient preservation, multiply by T² here and document the deviation.
```

Cohérent avec findings. ✅ Docstring OK, rien à changer.

#### (c) Numerical stability — `clamp_min(1e-12)`

Port ajoute `soft_targets.clamp_min(1e-12).log()` vs student `soft_targets.log()` direct.

**Justification port** : si un logit teacher est très négatif, `softmax` peut retourner `0.0` en
float32 (underflow), et `log(0) = -inf` → NaN propagation.

**Student** : aucun guard — mais en pratique les logits teacher sont rarement assez extrêmes pour
underflow softmax en float32.

**Divergence numérique port-student** : le clamp modifie **très légèrement** le gradient quand
`soft_targets` est proche de zéro (~1e-12). Effet négligeable sur training trajectories.

✅ **Gain port safe, divergence numérique négligeable.**

#### (d) Empty dead-code path

Port L166-167 :
```python
if hard_labels is None:
    return alpha * soft_loss
```

**Student** : ne court-circuite pas, fait `alpha * soft + (1-alpha) * 0 = alpha * soft`. Résultat
identique, juste différence de code style.

✅ **Equivalent sémantiquement.**

#### (e) Cross-reference `DynamicLossWeighter` (SARL+CL)

Le student SARL+CL utilise `DistillationLoss` via une dictionnaire de poids (`'distillation':
1.0` etc.). Le port refactore en callable direct. Les callers (`sarl_cl/trainer.py`) doivent donc
wrap explicitement via `DynamicLossWeighter` s'ils veulent parity — à vérifier en C.9 ou
Phase D SARL+CL.

**Note pour C.9** : cross-ref les 2 concepts `distillation_loss` (ici) vs
`weight_regularization` (L173-214) dans le même review. Le paper appelle les 2 "distillation"
(🆘 terminologie paper confuse).

#### Résumé `distillation_loss`

- ✅ **Parity student** (SARL+CL) bit-exact sauf le clamp_min (amélioration port).
- ⚠️ **D-003 existant** : pas de `T²` scaling (port = student, student = pas Hinton).
  Docstring port OK.
- ⚠️ **Student commentaire mensonger** (L394 SARL+CL) : dit "scaled by T²" mais ne l'est pas.
  Non-actionable pour nous, juste à noter.
- ✅ **Numerical safety** `clamp_min(1e-12)` : gain port safe.
- ⚠️ **Cross-ref `weight_regularization`** : terminologie paper confuse, à clarifier C.9.

### Fixes identifiées C.8

| ID        | Fix                                                                     | Scope                                       | Effort |
|:----------|-------------------------------------------------------------------------|---------------------------------------------|:------:|
| C8-fix-1  | Docstring `wagering_bce_loss` : incompatibilité `n_wager_units=2` raw logits | `losses.py:96-116` docstring           | 3 min  |
| C8-fix-2  | 🚨 Renommer `pos_weight` → `weight` (ou supprimer) pour matcher API `F.binary_cross_entropy` | `losses.py:90-123` + callers (none)   | 10 min |
| C8-fix-3  | (debate, skip) Default `reduction` : garder "mean" (PyTorch standard), documenter que callers passent "sum" | — | skip |
| C8-fix-4  | Docstring `distillation_loss` déjà OK (D-003 documenté) — rien à faire  | —                                           | skip |

### Cross-reference deviations.md

- **D-003 existant** : re-confirmé par cette review. Aucune mise à jour nécessaire.
- **Nouveau (optionnel)** : ajouter une note dans D-001 (ou D-005 nouveau) pointant vers
  C8-fix-1 — "`wagering_bce_loss` incompatible avec `n_wager_units=2`". Peut-être un D-005
  "wagering loss API coupling with WageringHead n_wager_units".

### Résumé — `wagering_bce_loss` + `distillation_loss`

- ✅ **Parity student** des 2 fonctions.
- ✅ **Math paper eq.5** respectée (BCE-on-probs ≡ BCE-with-logits).
- 🚨 **`pos_weight` mal nommé** → C8-fix-2 (renommer/supprimer, pas de caller affecté).
- ⚠️ **`wagering_bce_loss` + `n_wager_units=2`** incompatibles → C8-fix-1 (docstring warning).
- ⚠️ **D-003 T² scaling** : port fidèle au student qui diverge de Hinton.
- ✅ **Numerical safety** `clamp_min` : gain port.
- **0 divergence paper structurelle, 2 bugs API mineurs, 0 risque prod actuel.**

**C.8 clôturée. 2 fixes + 0 update deviations.md.**

---

---

## `weight_regularization` + audit callers losses.py (C.9)

### `weight_regularization` (L173-214) — paper's "distillation" L2 anchor

Port ~42 lignes :

```python
def weight_regularization(model, teacher_model):
    reg_loss = model.parameters().__next__().new_zeros(())  # scalar on the right device
    for param, param_teacher in zip(model.parameters(), teacher_model.parameters(), strict=True):
        reg_loss = reg_loss + torch.sum((param - param_teacher) ** 2)
    return reg_loss
```

#### (a) Conformité student

**Student** (`sarl_cl_maps.py:410-416`) :

```python
def compute_weight_regularization(model, teacher_model):
    reg_loss = 0
    for param, param_old in zip(model.parameters(), teacher_model.parameters()):
        reg_loss += torch.sum((param - param_old) ** 2)
    return reg_loss
```

**Mapping port ↔ student** :

| Step                   | Student                                       | Port                                                          | Match |
|------------------------|-----------------------------------------------|---------------------------------------------------------------|:-----:|
| Scalar init            | `reg_loss = 0` (Python int)                   | `model.parameters().__next__().new_zeros(())` (tensor on device) | ✅ (port meilleur — device coherence) |
| Param pairing          | `zip(model.params(), teacher.params())`       | `zip(..., strict=True)`                                       | ⚠️ port stricter (student silent on mismatch) |
| Per-param L2 diff      | `torch.sum((param - param_old) ** 2)`         | `torch.sum((param - param_teacher) ** 2)`                     | ✅    |
| Accumulate             | `reg_loss += ...`                              | `reg_loss = reg_loss + ...`                                   | ✅ (équivalent, in-place vs out-of-place)  |
| Return                 | scalar tensor                                  | scalar tensor 0-d                                             | ✅    |

✅ **Math identique**. 2 améliorations port (device init + strict zip) qui **durcissent** par
rapport au student sans changer le résultat numérique.

#### (b) ⚠️ `strict=True` — docstring incohérent

**Port L187-192 docstring** :
> ``teacher_model``. Must have the same parameter topology as ``model``; **we do not assert this**
> so the caller can pass e.g. networks sharing only a prefix — mismatches surface as shape errors
> from ``torch.sum``.

**Port L212 code** :
```python
for param, param_teacher in zip(model.parameters(), teacher_model.parameters(), strict=True):
```

Le `strict=True` **ASSERT** (ValueError si les itérables diffèrent en longueur). Le docstring
dit le contraire.

**Finding C9-F1** : docstring et code se contredisent. Le code est plus sécurisé (strict=True
raise immédiatement si topologies différentes) mais le docstring suggère que ça crash plus tard
via `torch.sum` shape error. Le code est **plus protecteur** que ce que le docstring promet.

**Impact caller** : `zip(..., strict=True)` raise sur différence de **longueur** d'itérable,
pas sur différence de shape. Un modèle avec moins de params passera bien la comparaison sur les
premiers params si même topo — donc la promesse du docstring ("caller peut passer prefix match")
est **fausse** : strict=True empêche.

**Piste C9-fix-1** : corriger le docstring — soit (a) aligner sur le code (strict length check,
runtime ValueError), soit (b) enlever `strict=True` et vraiment laisser les shape errors apparaître
en `torch.sum`. Recommandation : **garder `strict=True` et corriger docstring** (version (a),
sécurité > flexibilité mal documentée).

#### (c) Gradient flow — teacher doit être frozen

Le port **n'assert pas** que `teacher_model.parameters()` sont `requires_grad=False`. Si un
caller oublie de freeze teacher :

```python
# Caller wrong-pattern:
teacher_first_net.load_state_dict(policy_net.state_dict())  # copy weights
# forgot: teacher_first_net.requires_grad_(False)
loss = weight_regularization(policy_net, teacher_first_net)
loss.backward()  # gradient flows to teacher too → teacher drifts!
```

Le terme `(param - param_teacher)**2` dépend des 2 params, donc backward flow dans les 2 sans
teacher frozen. Bug silencieux : teacher drift au lieu d'anchor statique.

**Student** : même piège, pas d'assertion.

**Callers actuels port** (`sarl_cl/trainer.py:284, 329`) : à vérifier que teachers sont
correctement frozen.

**Piste C9-fix-2** : (info) ajouter une mention explicite dans docstring "CALLER MUST call
`teacher_model.requires_grad_(False)` before passing". Runtime assert possible mais coûteux à
chaque appel → skip.

#### (d) Naming divergence — "distillation" (paper) ≠ distillation_loss (port)

Le docstring port L205-209 est **lucide** sur ce point :

> The paper calls this "distillation" but strictly speaking it is an L2 regularization anchor,
> not output-distillation. We keep the ``distillation`` key in ``DynamicLossWeighter`` for
> faithful parity with the paper's dictionary keys, but code-level we use
> ``weight_regularization`` to avoid confusion with :func:`distillation_loss` above.

**Résolution terminologie paper-vs-port** :

| Concept                     | Paper terme      | Port terme                                     | Utilisé où ?                              |
|:----------------------------|:-----------------|:-----------------------------------------------|:------------------------------------------|
| KL soft-target (Hinton)     | "distillation"   | `components.losses.distillation_loss`         | **Jamais appelé** en prod (voir audit C.9-e) |
| L2 param anchor             | "distillation"   | `components.losses.weight_regularization`     | `sarl_cl/trainer.py:284, 329`              |
| Dict key                    | `'distillation'` | `'distillation'` (via `DynamicLossWeighter`)   | weight dict (paper fidelity)              |

✅ **Résolution port claire** : 2 noms code distincts pour 2 concepts, mais clé dict paper
préservée. Faithful et lisible.

### (e) Audit final — callers losses.py

Grep exhaustif sur `src/` :

| Symbol                                    | Callers                                          | Status |
|:------------------------------------------|:-------------------------------------------------|:------:|
| `components.losses.cae_loss`              | `blindsight/trainer.py:329`, `agl/trainer.py:300` | ✅    |
| `sarl.losses.cae_loss`                    | `sarl/trainer.py:188`, `sarl_cl/trainer.py:283, 302` | ✅    |
| `components.losses.wagering_bce_loss`     | `blindsight/trainer.py:311`, `agl/trainer.py:282` | ✅    |
| `components.losses.distillation_loss`     | **0 callers** (exposé via `__init__.py` seulement) | ⚠️ dead code |
| `components.losses.weight_regularization` | `sarl_cl/trainer.py:284, 329`                    | ✅    |

**Finding C9-F2 — `distillation_loss` est du dead code** :

- Exposé via `maps/components/__init__.py:6, 22` ("distillation_loss" public API).
- **Aucun caller** en prod (SARL+CL utilise `weight_regularization`, pas KL distillation).
- Le paper DEFINIT une `DistillationLoss` class (student `sarl_cl_maps.py:359-405`) mais **ne
  l'appelle jamais** non plus dans le training loop.
- Documenté dans `docs/reports/sprint-04b-report.md:37` :
  > *"The paper defines a `DistillationLoss` class but never calls it; the actual anchor in
  > `train()` is `compute_weight_regularization`."*

**Conséquence** : `distillation_loss` est **un port fidèle d'un code mort**. Le garder :
- ✅ préserve la parity avec student (qui a la même classe inutilisée).
- ⚠️ peut induire en erreur (un lecteur pense que SARL+CL fait de la distillation KL).
- ⚠️ coût maintenance : tests à garder verts, signature à maintenir.

**Piste C9-fix-3** : ajouter un commentaire `"# ⚠️ Not used in production — see docs/reports/
sprint-04b-report.md:37"` au-dessus de `def distillation_loss` L126. Garder l'import exposé pour
éventuelle utilisation future. **Pas de suppression** (user policy "preserve everything that
reproduces paper").

**Finding C9-F3 — duplication `sarl.losses.cae_loss`** :

Rappel C.7 : `components.cae_loss` (BCE/MSE) et `sarl.losses.cae_loss` (Huber) coexistent.
Audit confirme **aucun cross-call** : Blindsight/AGL utilisent `components`, SARL/SARL+CL utilisent
`sarl.losses`. Pas de crossover accidentel. DETTE-2 déjà tracé en C.7.

### (f) Signature callers — vérification exhaustive args

**`cae_loss`** (Blindsight/AGL version) :
```python
cae_loss(W=self.first_order.fc1.weight, x=..., recons_x=..., h=..., lam=..., recon="bce_sum")
```
✅ Tous les callers passent `recon="bce_sum"` explicitement. Pas de piège default.

**`sarl.losses.cae_loss`** :
```python
cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)  # positional
```
✅ 5 positional args correspondent à la signature. `CAE_LAMBDA` constante module-level.

**`wagering_bce_loss`** :
```python
wagering_bce_loss(wager.squeeze(-1), target, reduction="sum")       # Blindsight
wagering_bce_loss(wager, target, reduction="sum").requires_grad_()  # AGL
```
⚠️ AGL appelle `.requires_grad_()` post-loss — force gradient tracking explicite. Probablement
legacy (PyTorch gère auto). Pas un bug mais noise.

**Piste C9-fix-4** (cosmétique, optionnel) : vérifier si `.requires_grad_()` L282 AGL est
nécessaire. Si non (probable), retirer pour clarté. Hors scope C.9, à faire en Phase D.

**`weight_regularization`** :
```python
weight_regularization(policy_net, teacher_first_net)        # main
weight_regularization(second_order_net, teacher_second_net) # meta
```
✅ Signature correcte. Teacher frozen ? À vérifier dans `sarl_cl/trainer.py` — **audit hors C.9
scope**, on flagge.

### Fixes identifiées C.9

| ID        | Fix                                                                    | Scope                                  | Effort |
|:----------|------------------------------------------------------------------------|----------------------------------------|:------:|
| C9-fix-1  | Docstring `weight_regularization` : corriger incohérence `strict=True` vs "we do not assert" | `losses.py:190-195` docstring | 5 min  |
| C9-fix-2  | Docstring `weight_regularization` : explicite "caller MUST freeze teacher via `requires_grad_(False)`" | `losses.py:200-202` docstring | 3 min  |
| C9-fix-3  | Commentaire L126 `distillation_loss` : "dead code in prod, kept for future" | `losses.py:126` inline comment     | 2 min  |
| C9-fix-4  | (skip C.9) vérifier `.requires_grad_()` AGL L282 — Phase D scope       | `agl/trainer.py:282`                   | hors  |

### Cross-reference deviations.md

- **Paper terminology "distillation"** : déjà documenté dans `docs/reports/sprint-04b-report.md:37`.
  Pas besoin d'ajouter à `deviations.md`, c'est un **notre-choix-de-naming**, pas une divergence.
- **`distillation_loss` dead code** : à mentionner comme `DETTE-3` candidate (code mort port
  fidèle au student). Optionnel — C.10 decision.

### Résumé `weight_regularization`

- ✅ **Math parity student** bit-exact (L2 param diff, sum over params).
- ✅ **Port améliorations safe** : tensor init + `strict=True`.
- ⚠️ **Docstring incohérent** avec code (strict claim) → C9-fix-1.
- ⚠️ **Teacher-frozen responsibility silencieuse** → C9-fix-2.
- ✅ **Naming divergence paper resolved** (docstring L205-209 claire).

### Résumé audit callers losses.py (C.9)

- ✅ **6 callers au total** sur `components.losses` + `sarl.losses`, tous cohérents avec
  signatures.
- ⚠️ **`distillation_loss` dead code** → C9-fix-3 (commentaire) + DETTE-3 candidate.
- ⚠️ **`.requires_grad_()` AGL L282** à investiguer Phase D.
- ✅ **DETTE-2** (duplication `cae_loss`) : pas de crossover accidentel confirmé.
- ✅ **Terminology paper "distillation"** : résolu côté port (2 noms code, 1 dict key préservé).

**C.9 clôturée. 3 fixes + 1 DETTE candidate + 1 Phase D note. Review losses.py complète.**

---

## Résumé global C.7 + C.8 + C.9 — losses.py review

| Sub-phase | Scope                                              | Findings clés                                       | Fixes C.10  |
|:---------:|----------------------------------------------------|-----------------------------------------------------|:-----------:|
| C.7       | `components.cae_loss` + D-002 deep dive           | 🚨 D-002 SimCLR vs CAE réelle ; DETTE-2 duplication ; docstring SARL erroné state_dict | 5           |
| C.8       | `wagering_bce_loss` + `distillation_loss`          | 🚨 `pos_weight` mal nommé ; incompat `n_wager_units=2` ; D-003 re-confirmé ; student comment ment | 2 |
| C.9       | `weight_regularization` + audit callers            | ⚠️ docstring strict inconsistency ; dead code `distillation_loss` ; naming paper-faithful | 3 (+1 DETTE) |

**Total fixes pour C.10 batch** : 10 fixes + 2 DETTEs + 1 sub-phase D.17 à créer (décision
SimCLR vs CAE).

**État `losses.py`** : **entièrement reviewé, 0 bug critique bloquant, 2 bugs API mineurs
(`pos_weight`, docstring strict), 1 divergence structurelle paper (D-002), 2 dettes (duplication +
dead code).**

**Next**: C.10 — apply fixes (similaire à C.6 pour C.3-C.5 batch).

