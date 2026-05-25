# Reverse-prompt — `src/maps/core/losses.py`

**Module path actuel (sur main) :** `src/maps/components/losses.py`
(renommé `core/losses.py` en Sprint 11+).
**Taille :** 215 lignes, 4 fonctions publiques.
**Paper :** §2.1 eq.4 (contrastive — voir D-002), eq.5 (BCE wager),
§2.2-§5 (distillation CL, Table 11). Eq.17 (CL weighted).
**Référence externe :** Rifai et al. (2011) CAE, Hinton et al. (2015)
KD, Chen et al. (2020) SimCLR (mentionné mais non implémenté).
**Review existant :** `docs/reviews/losses.md` (Sprint-08 C.7/C.8/C.9,
~720 lignes — très exhaustif).

⚠️ **Module doublé** : il existe AUSSI `src/maps/experiments/sarl/losses.py`
(version Huber pour SARL/SARL+CL). DETTE-2.

---

## 1. Quelle aurait été la spec ?

**Prompt qu'on aurait donné à Claude :**

> Écris un module Python `losses.py` qui implémente les 4 fonctions de
> loss du paper MAPS :
>
> **1. `cae_loss(W, x, recons_x, h, lam, *, recon="bce_sum")`**
> Contractive AutoEncoder loss (Rifai et al. 2011) :
>     L_CAE = recon(x, x̂) + λ · ||J_h(x)||²_F
> Le terme contractif utilise la forme analytique sigmoid-derivative
> `h(1-h)` × `Σ_i W_ji²` (somme sur les colonnes de W²).
> **Note importante** : le paper §2.1 eq.4 décrit une SimCLR contrastive
> loss (Chen 2020), pas un CAE. Mais le code student de Juan utilise CAE
> (Rifai 2011). Comme les Tables 5/6/7 paper ont été produites par le
> code student, on garde CAE pour reproduction. SimCLR variant à
> exposer en stub `NotImplementedError`. Voir D-002.
> Le param `recon` permet "bce_sum" (Blindsight/AGL — `recons_x` doit
> être post-sigmoid ∈ [0,1]), "mse_mean", ou "mse_sum".
> Détacher W avant le calcul du Jacobian term (matche `state_dict()`
> default `keep_vars=False` du student → gradient ne flow PAS directement
> vers W, seulement via `h`).
>
> **2. `wagering_bce_loss(wager, target, reduction="mean")`**
> BCE on probabilities (matches student `nn.BCELoss(size_average=False)`).
> **Pas BCE-with-logits** : prend des sigmoid outputs en input. C'est
> équivalent mathématiquement mais moins numériquement stable.
> Incompatible avec `WageringHead(n_wager_units=2)` qui retourne raw
> logits — caller doit utiliser `F.binary_cross_entropy_with_logits`
> directement dans ce cas.
>
> **3. `distillation_loss(student_logits, teacher_logits, hard_labels=None,
> alpha=0.5, temperature=2.0)`**
> Hinton KL distillation (soft + optional hard CE). **D-003** : le paper
> implicite et Hinton 2015 recommandent scaling par T² pour préserver
> les gradient magnitudes — student code ne le fait pas (et le
> commentaire student dit "scaled by T²" mais ment). On garde fidèle au
> student. ⚠️ Cette fonction n'est **jamais appelée en prod** (SARL+CL
> utilise `weight_regularization` à la place) — DETTE-3 candidate.
>
> **4. `weight_regularization(model, teacher_model)`**
> L2 param drift anchor : `Σ_i (θ_i - θ_i^teacher)²`. Le paper appelle
> ça "distillation" dans `DynamicLossWeighter` dict keys mais c'est
> en réalité une régularisation EWC-style (Kirkpatrick 2017) sans
> Fisher weighting. Caller DOIT freezer le teacher
> (`teacher.requires_grad_(False)`) sinon gradient drift silencieux.
> Use `zip(..., strict=True)` pour fail-fast si topologies diffèrent.

## 2. Contraintes scientifiques

### Équation paper eq.4 (contrastive, **NON implémentée**)

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i}^{2N} \exp(\text{sim}(z_i, z_k)/\tau)}$$

C'est la **NT-Xent loss de SimCLR (Chen et al. 2020)** :
- Variables `z_i, z_j` = paires positives (augmentations du même sample)
- 2N pairs dans le batch
- Cosine similarity `sim`, température `τ`

**Le student utilise un Contractive AutoEncoder à la place** :

$$\mathcal{L}_{CAE} = \text{recon}(x, \hat{x}) + \lambda \|J_h(x)\|^2_F$$

avec forme analytique sigmoid : $\|J_h\|^2_F = \sum_j (h_j(1-h_j))^2 \sum_i W_{ji}^2$.

**Les deux losses sont mathématiquement distinctes** (objectif, gradient,
hyperparams). Voir D-002 deep dive ci-dessous.

### Équation paper eq.5 (BCE wagering)

$$\mathcal{L}_{BCE} = -[y \log \sigma(W) + (1-y) \log(1 - \sigma(W))]$$

Implémentée bit-exact via `binary_cross_entropy` sur probs (sigmoid
pré-appliqué dans `WageringHead`) — équivalent mathématiquement à
BCE-with-logits.

### Citations paper

- **Rifai et al. (2011)** : "Contractive auto-encoders". CAE original.
  Formule `h(1-h)` valide pour encoder sigmoid (dérivée sigmoid).
- **Hinton et al. (2015)** : "Distilling the knowledge in a neural
  network". Soft targets, temperature scaling T². T² scaling **non
  implémenté** par student.
- **Chen et al. (2020)** : SimCLR / NT-Xent. Mentionné paper §2.1 mais
  non implémenté.
- **Kirkpatrick et al. (2017)** : EWC. Référence pour
  `weight_regularization` (simplification : pas de Fisher weighting).

### Hyperparamètres

- **λ_CAE = 1e-4** (paper §2.1 et `maps.yaml`)
- **Distillation α = 0.5, T = 2.0** (paper §2.2, Table 11, `maps.yaml`)
- **CL weights `(weight_task, weight_distillation, weight_feature) =
  (0.4, 0.4, 0.2)`** (paper text p.17 "optimal", Figure 7 — `(0.3,
  0.6, 0.1)` per Table 11 est une déviation D-cl-weights)

## 3. Contraintes d'ingénierie

### Quirk paper-faithful : `h(1-h)` sur ReLU output

La forme `h(1-h)` est la dérivée de **σ(z)·(1-σ(z))** — valide quand
`h = σ(z)`. **Tous les 3 domaines (Blindsight, AGL, SARL) utilisent
ReLU encoder**, donc la formule est **mathématiquement fausse** (dérivée
de ReLU = `1[z > 0]`, pas `h(1-h)`).

Student le fait dans les 3 domaines → notre port le préserve byte-for-byte
pour parity. **C'est un quirk universel, pas un bug à corriger sans
discussion**.

### Détacher W ou pas ?

`W_const = W.detach()` dans le port. Pourquoi : le student accède à W
via `state_dict()['fc1.weight']`, qui par défaut (`keep_vars=False`)
retourne un tensor détaché. Donc gradient **ne flow PAS** directement
vers W via le terme contractif — seulement via `h`'s backward path.

Comportement subtil mais paper-faithful via matching student.

### Duplication `cae_loss` (DETTE-2)

| Aspect | `core.losses.cae_loss` | `sarl/losses.cae_loss` |
|--------|------------------------|------------------------|
| Reconstruction | BCE / MSE | Huber (hardcodé) |
| Encoder activation supposée | Sigmoid (formule correcte) | ReLU (formule préservée malgré fausseté) |
| Callers | Blindsight, AGL trainers | SARL, SARL+CL trainers |

Justifications de la duplication :
- Reconstruction terms hétérogènes
- Activations différentes (mais formule `h(1-h)` partout — incohérent
  avec la justification "activations différentes" en réalité)

**Quand unifier** : post-Phase F. Voie :
`cae_generic(W, x, recons_x, h, lam, recon_fn, hidden_activation)` ou
`cae_loss(recon="bce_sum" | "huber" | "mse_*")`.

### `distillation_loss` = dead code (DETTE-3)

`grep` confirmé : **0 callers en prod**. Le student SARL+CL définit la
classe `DistillationLoss` mais ne l'invoque jamais. Le vrai "distillation"
en CL est `weight_regularization` (L2 anchor sans Fisher).

Le port le garde par fidélité student. Mais c'est du code mort qui peut
induire en erreur. Recommandation : commentaire `⚠️ Not used in production`
et possiblement demote à `_distillation_loss` private.

### `wagering_bce_loss` paramètre mal nommé (bug API)

Le param `pos_weight` est passé à `F.binary_cross_entropy` comme
`weight=` — qui attend **per-sample weights**, pas per-class. Le nom
suggère class imbalance (API `BCEWithLogitsLoss.pos_weight`) mais
sémantiquement c'est différent. 

**Callers actuels** : aucun n'utilise `pos_weight`, donc bug latent
non-déclenché. **À renommer en `weight` ou supprimer si YAGNI** (Sprint
11+).

## 4. Deviations / dettes présentes

### 🚨 D-002 — Contrastive (paper) vs CAE (student/port)

**Divergence structurelle paper-vs-student**. Paper §2.1 eq.4 décrit
SimCLR (Chen 2020), code student implémente CAE (Rifai 2011). Les 2
losses sont mathématiquement disjointes (objectif, gradient, hyperparams).

**Sprint-08 décision D.22b** : garder CAE par défaut (reproduit Tables
5/6/7 student), exposer SimCLR comme stub `NotImplementedError`. Si
Phase F manque les z-scores headline par une marge attribuable à la
loss, revisiter.

Implication : les chiffres paper Tables 5/6/7 ne peuvent **pas** avoir
été produits par le code student (CAE) avec la prose paper (SimCLR).
Soit (a) une version du code différente a produit les chiffres, soit
(b) les chiffres paper sont incorrects sur ce point.

### ⚠️ D-003 — Distillation T² scaling

Hinton (2015) recommande `loss_soft *= T²` pour préserver gradient
magnitudes. Student ne le fait pas (et le commentaire student L394 ment
explicitement : "scaled by T²" alors qu'il ne l'est pas). Notre port
garde l'absence de scaling pour parity. Gradient ~1/T² plus petit
qu'attendu — absorbé par α weighting ou learning rate dans les runs
student.

### ⚠️ D-004 — AGL chunked sigmoid (lié au decoder, pas à `losses.py`)

Mentionné pour traçabilité : AGL applique sigmoid par chunk de 6 bits
(`make_chunked_sigmoid(6)`). Affecte `recons_x` pour `cae_loss`. Pas
une déviation côté `losses.py`.

### DETTE-2 — Doublon `cae_loss`

(voir Contraintes d'ingénierie §3 ci-dessus)

### DETTE-3 — `distillation_loss` dead code

(voir Contraintes d'ingénierie §3 ci-dessus)

### Quirks docstring résolus en Sprint-08

- 🚨 Bug `n_wager_units=2` softmax → fixed C.6 (raw logits)
- ⚠️ Docstring `weight_regularization` strict claim mismatch → C9-fix-1
- ⚠️ Caller-must-freeze-teacher silencieux → C9-fix-2 docstring update

## 5. Questions ouvertes

- **Q1 :** D-002 est la déviation la plus structurelle de tout le projet.
  Le paper dit SimCLR, le code dit CAE. **Quelqu'un parmi les auteurs
  (Juan, Antoine, Axel, Zahra) a forcément remarqué** au moment de
  l'écriture. Pourquoi avoir maintenu cette divergence sans la
  documenter ? Question sociale autant que technique.

- **Q2 :** Le `h(1-h)` sur ReLU output est universel dans le code
  student. Est-ce :
  - (a) un copier-coller du code CAE original de Rifai (formule sigmoid)
        sans s'apercevoir que l'encoder a été changé ?
  - (b) un choix délibéré (peut-être la formule "incorrecte" fonctionne
        mieux que la dérivée ReLU correcte) ?
  Ablation : remplacer par `(h > 0).float()` et comparer Tables 5/6.

- **Q3 :** `distillation_loss` dead code — peut-on le supprimer en
  toute confiance ou bien faut-il garder pour fidélité student ? Le
  garder pollue l'API. Le supprimer perd la fidélité 1-to-1 mais ne
  change rien aux outputs.

- **Q4 :** *(à toi, Rémy)* — qu'est-ce qui te paraît non-obvious ?

## 6. Pour le rewrite (Sprint 11+)

### À garder tel quel
- API `cae_loss(W, x, recons_x, h, lam, *, recon="bce_sum")`
- Quirk `h(1-h)` sur ReLU output (paper-faithful sur les 3 domaines)
- `W.detach()` (matche student `state_dict()` default)
- `distillation_loss` math (KL formule, clamp_min(1e-12) numerical safety)
- `weight_regularization` math + `strict=True` zip

### À améliorer / décider pour Sprint 11+

- **Unification `cae_loss` (DETTE-2)** : merger en une fonction avec
  `recon="bce_sum" | "huber" | "mse_*"`. Le seul vrai différentiateur
  entre les 2 versions actuelles est le reconstruction term.
- **`distillation_loss` (DETTE-3)** : soit demote à private `_distillation_loss`
  (signal "non utilisé"), soit supprimer carrément. À trancher.
- **`wagering_bce_loss.pos_weight`** : renommer `weight` (matches
  `F.binary_cross_entropy`) ou supprimer (YAGNI, 0 callers actuels).
- **SimCLR stub** : implémenter `simclr_loss` qui raise
  `NotImplementedError` en signalant la marche à suivre (augmentations,
  τ, NT-Xent formule). Préparer pour un futur où on voudrait le
  paper-faithful path.
- **Docstring `cae_loss`** : être explicite sur quirk `h(1-h)` ReLU
  (C7-fix-2).

### Tests à écrire (avant de toucher au code)
- **Test parity cae_loss vs student `blindsight_tmlr.py:91-129`** : 
  même W, x, recons_x, h, λ → même valeur scalaire à 1e-6 près
- **Test parity `wagering_bce_loss`** : match `nn.BCELoss(size_average=False)`
- **Test `weight_regularization`** : sur 2 modules identiques → 0 ;
  sur teacher non-frozen → vérifie que gradient flow vers teacher
  (et donc qu'on doit assert freeze côté caller)
- **Test `distillation_loss` clamp** : avec logits teacher très négatifs
  (1e-20 softmax), pas de NaN
- **Test `cae_loss` recon variants** : `bce_sum` vs `mse_mean` produisent
  des valeurs cohérentes
- **Test invariant `W.detach()`** : gradient backward ne touche pas W

### Opportunités perf identifiées
- **`F.binary_cross_entropy_with_logits`** au lieu de
  `F.binary_cross_entropy(sigmoid(x))` quand `n_wager_units=2` : plus
  stable numériquement, pareil mathématiquement
- **CAE Jacobian via `einsum`** : `torch.einsum('bh,hi->bi', dh**2,
  W**2)` pourrait être marginalement plus rapide que `torch.mm` (à
  bencher, gain négligeable probablement)
- **Cache `W**2` row sum** quand on appelle `cae_loss` plusieurs fois
  avec le même W (boucle cascade par exemple) : actuellement recalculé
  à chaque appel

### Connexion avec d'autres modules
- `cae_loss` appelé par : `domains/blindsight/trainer.py`,
  `domains/agl/trainer.py` (via core.losses), `domains/sarl/trainer.py`,
  `domains/sarl_cl/trainer.py` (via sarl/losses — DETTE-2)
- `wagering_bce_loss` appelé par : `domains/blindsight/trainer.py`,
  `domains/agl/trainer.py`
- `distillation_loss` : 0 callers (dead code)
- `weight_regularization` : `domains/sarl_cl/trainer.py:284, 329`

## Méta — pour une session Claude chat

Prompt prêt à coller dans Claude chat (web) :

```text
J'étudie l'implémentation de la "first-order loss" dans le paper MAPS
(Vargas et al. 2025). Le paper §2.1 eq.4 décrit une SimCLR / NT-Xent
contrastive loss (Chen et al. 2020) :

    L = -log[exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ)]

Mais le code student implémente un Contractive AutoEncoder (Rifai et al.
2011) :

    L = recon(x, x̂) + λ ||J_h(x)||²_F

Ce sont deux losses **mathématiquement distinctes** (objectif, gradient,
hyperparams). Le code student écrit dans ses commentaires "contrastive
learning" mais implémente CAE.

Trois questions :
1. Y a-t-il un lien conceptuel entre CAE et SimCLR que je rate ? Les
   deux régularisent les représentations latentes mais par des
   mécanismes très différents.
2. Si on devait implémenter SimCLR pour ce projet, quelles paires
   positives utiliser ? Le domaine Blindsight a des stimulus binaires
   (0/1 patterns), AGL a des sequences 8-letter binaires. Les
   augmentations standard SimCLR (crop, color jitter) ne s'appliquent
   pas.
3. Y a-t-il dans la littérature un travail qui compare CAE vs SimCLR
   sur des tâches de classification supervisée avec représentations
   intermédiaires ? J'aimerais avoir un ordre d'idée du gap qu'on
   verrait en reproduisant les Tables paper avec l'une ou l'autre.
```
