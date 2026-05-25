# Reverse-prompt — `src/maps/domains/sarl_cl/loss_weighting.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl_cl/loss_weighting.py`
**Taille :** ~190 lignes.
**Paper :** §4 (continual learning loss composition).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `loss_weighting.py` pour la normalisation
> dynamique des **3 termes de loss CL** :
>
> - **`task`** : loss de la tâche courante (CAE pour FO, BCE pour SO)
> - **`distillation`** : L2 weight regularization vs teacher
>   (le nom "distillation" est paper-faithful mais misleading — c'est
>   pas du Hinton KL distillation, c'est de l'EWC sans Fisher weighting)
> - **`feature`** : MSE entre activations intermédiaires student et
>   teacher (h1 pour FO, comparison_out pour SO)
>
> Ces 3 termes vivent sur des scales différentes :
> - CAE ~ 1e-2
> - weight_regularization peut atteindre 1e+1
> - feature MSE varie
>
> Naïvement summing → un term domine le gradient. Le paper normalise
> par le **historical maximum** de chaque term + epsilon.
>
> **Classes / fonctions** :
>
> **1. `DynamicLossWeighter`** :
> - `__init__(update_interval=10_000, keys=("task", "distillation",
>   "feature"))`
> - `update(losses)` : detach scalar values, track running max
> - `weight_losses(losses, epsilon=1e-16)` : retourne `{k: losses[k] /
>   (historical_max[k] + ε)}`
> - `get_stats()` : diagnostic snapshot
>
> **2. Helpers** (paper-faithful exposés pour cohérence API) :
> - `update_moving_average(current, new, momentum=0.9)` : EMA standard
>   (jamais appelé par le weighter actuel, kept pour parity API)
> - `min_max_norm(mean, values)` : paper helper avec nom trompeur —
>   divise `mean / max(values)`, pas vraie normalization min-max
> - `individual_losses(output, target, loss_fn)` : 95th percentile de
>   loss per-sample (analyse exploratoire paper)
>
> **Note critique** : le paper original utilisait EMA pour
> normalization. La version finale (que le port reproduit) utilise
> **running max**. L'EMA path est commenté out (lines 579-586 source).
> Notre port garde l'helper public pour parity API + flexibility.

## 2. Contraintes scientifiques

### Pourquoi normalize ?

Si on combine `α·loss_task + β·loss_distill + γ·loss_feat` avec α,β,γ
constants, les magnitudes naturelles dominent. Si distill = 100·task,
β doit être tiny pour ne pas dominer. Mais en début de training,
distill peut être 0 (params identiques teacher = student initialement)
→ β devrait être grand. Conflict.

**Running max normalization** :
`loss_normalized = loss / max_seen_so_far` → toujours dans [0, 1] à
peu près. Permet d'utiliser des `α, β, γ` constants sensés.

### Final weights (D-cl-weights D.20)

Sprint-08 D.20 (puis update Sprint-08 Phase F.4 reversal) : weights
canoniques = **(0.4, 0.4, 0.2)** paper text p.17 "optimal" + Figure 7.

Paper Table 11 dit (0.3, 0.6, 0.1). Paper text dit (0.4, 0.4, 0.2).
Student code utilise (1.0, 1.0, 1.0).

Choix retenu : (0.4, 0.4, 0.2) — paper text p.17 dans "optimal weights"
+ Figure 7 IS the headline CL result. Tableau Table 11 serait moins
canonique que la prose narrative.

### EWC parallel (Kirkpatrick 2017)

L'L2 weight anchor `Σ (θ_i - θ_i^teacher)²` est une **EWC simplifiée**
sans Fisher information weighting. Vrai EWC pondère chaque param par
sa "importance" (Fisher info). Notre version pondère tous les params
égalitairement.

Simplification deliberate (paper choix), pas un bug.

## 3. Contraintes d'ingénierie

### Running max accumulate over training

Important : le `historical_max` ne reset PAS. Donc sur des longs runs,
les normalizations deviennent stables (les peak losses initiaux
restent la référence).

Si on voulait un running max "récent" (window-based), il faudrait
implémenter. Pour CL ça pose problème : le max changera selon le
curriculum (chaque task a son propre peak), donc "lifetime running max"
peut être dominé par early-task peaks.

C'est probablement OK car la normalisation est principalement pour
éviter le scale-imbalance initial, pas pour adapter dynamiquement.

### `update_interval` vestigial

Le paper source a `if self.steps % self.update_interval ==
self.update_interval // 2: historical_max_prev = ...`. Mais le
`historical_max_prev` n'est jamais lu par `weight_losses`. **Vestigial
code** kept pour parity.

### EMA helper exposé non-utilisé

`update_moving_average` est dans l'API mais pas appelée. Permet à un
caller de switcher en EMA-based weighting si besoin. Sans loss
significatif (40 lignes de dead-but-parity code).

### `min_max_norm` mal nommé

Fait `mean / max(values)`, pas (`val - min) / (max - min)`. Paper
quirk. Kept verbatim.

## 4. Deviations / dettes présentes

### D-cl-weights (Phase F.4 reversal resolved)

Voir §2. Weights = (0.4, 0.4, 0.2) paper text p.17 / Figure 7.

### Pas de DETTE active

Module propre. Helpers vestigials kept pour parity API.

## 5. Questions ouvertes

- **Q1 :** Le `historical_max` ne reset jamais. Sur un curriculum
  4-games, le peak du first game domine toute la normalization. C'est
  intentionnel ?

- **Q2 :** Pourquoi `update_interval=10_000` si jamais utilisé pour
  reset ? Confusing. Faut-il le supprimer ou implémenter le reset
  qu'il suggère ?

- **Q3 :** Comparaison rigoureuse running-max vs EMA vs running-mean :
  ablation jamais faite (à ma connaissance). Lequel est mieux ?

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- `DynamicLossWeighter` running-max algorithm
- 3 keys `("task", "distillation", "feature")` standard
- `epsilon=1e-16` numerical safety
- Helpers vestigials (`update_moving_average`, `min_max_norm`,
  `individual_losses`) pour parity API

### À améliorer
- **Documenter `update_interval` vestigial** explicitly
- **Pydantic schema pour `get_stats()` return**
- **Test alternative : EMA-based weighting** (ablation Sprint 11+)

### Tests à écrire
- `weight_losses` divise par historical max
- `update(losses)` track les peaks correctement
- Multi-step : peaks accumulate (jamais reset)
- `epsilon` safety : `max=0` → divisé par ε, pas NaN

### Connexion
- Utilisé par : `domains/sarl_cl/trainer.sarl_cl_update_step`
- Per-network instance (1 pour FO, 1 pour SO)
