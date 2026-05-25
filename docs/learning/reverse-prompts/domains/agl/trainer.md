# Reverse-prompt — `src/maps/domains/agl/trainer.py`

**Module path actuel (sur main) :** `src/maps/experiments/agl/trainer.py`
**Taille :** 819 lignes — le plus gros module AGL.
**Paper :** §4 + §A.2, Table 5b/5c, Table 10.
**Review existant :** `docs/reviews/agl-trainer.md`,
`docs/reviews/rg003-resolution.md`,
`docs/reviews/rg003-ablation-sweep.md`.

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `trainer.py` qui orchestre le full 3-phase
> protocole AGL paper §A.2. Porte `AGL/AGL_TMLR.py:547-1270` du student.
>
> **Phases** :
> 1. **Pretrain** (n_epochs_pretrain, random grammar) → trainer
>    learns 1st-order reconstruction + 2nd-order wager prediction
> 2. **Reset first-order** to initial weights (paper L751) — c'est ce
>    qui crée le contrast High/Low awareness via different fine-tuning
> 3. **Replicate** post-pretrain into 20-cell pool (via
>    `AGLNetworkPool`)
> 4. **Training Grammar-A** : High tier 12 epochs, Low tier 3 epochs
> 5. **Test** : Grammar-A + Grammar-B mix, per-cell + aggregated metrics
>
> **Classes** :
> ```python
> @dataclass(frozen=True)
> class AGLSetting:
>     # same 6-cell schema as BlindsightSetting
>     id, label, cascade_1st, cascade_2nd, second_order
>     @property cascade -> bool  # legacy alias
>     @classmethod from_dict  # 6-cell or legacy 2x2
>
> class AGLTrainer:
>     def __init__(cfg, setting, *, device="cpu")
>     def build()
>     def pre_train(n_epochs=None, batches=None) -> (losses_1, losses_2)
>     def training(n_epochs, batches=None) -> (losses_1, losses_2, precision)
>     def evaluate(eval_patterns_number=None, threshold=None) -> dict
>     def evaluate_pool(pool, eval_patterns_number=None, threshold=None) -> dict
> ```
>
> **Configuration optimizer** (paper Table 10) :
> - `RANGERVA` preferred — `torch_optimizer.RangerVA`
> - Fallback `ADAMAX` with warning si `torch-optimizer` non installé
> - Scheduler `StepLR(step_size=1, gamma=0.999)` (Table 10)
>
> **`train_meta_frozen_in_training=True`** (default, student L969) :
> pendant la phase Grammar-A, le 2nd-order **forward** mais ne fait
> PAS backward/step. Ablation D.28.h A1 confirme : si `False`, MAE
> dégrade de 0.014 à 0.025.
>
> **Reference reset behavior** (paper §A.2 L751) :
> `pre_train()` restore les **initial** first-order weights à la fin.
> C'est essentiel pour le mécanisme High/Low awareness — le 1st-order
> doit re-apprendre Grammar-A from scratch dans la phase suivante.
> Blindsight ne fait pas ce reset.
>
> **`_run_training_loop` extrait** (module-level) — partagé entre
> `AGLTrainer.training` (un cell) et `AGLNetworkPool.train_range` (N
> cells).

## 2. Contraintes scientifiques

### Pourquoi reset first-order après pretrain ?

C'est subtil. Pendant le pretrain :
- First-order apprend à reconstruire des mots random
- Second-order apprend à wager sur le succès du 1st-order

Si on gardait le 1st-order entraîné, il serait déjà "bon" sur les
mots random. Quand on training sur Grammar-A, il convergerait
rapidement → pas de différence High/Low.

En reset le 1st-order, on s'assure qu'il **commence from scratch** sur
Grammar-A. Le 2nd-order (gardé) sait déjà "comment juger la qualité
d'une reconstruction". L'écart de 12 epochs vs 3 epochs Grammar-A
crée le contrast d'awareness.

C'est l'essence du paradigme AGL Dienes : implicit vs explicit
knowledge transfer.

### Reference reset bit-exact

Implémentation : cache `self._initial_first_order_state =
copy.deepcopy(self.first_order.state_dict())` dans `.build()`. Restore
à la fin de `pre_train()` via `load_state_dict()`. Matches student
L751 byte-for-byte.

### `train_meta_frozen_in_training` ablation

Sprint-08 D.28.h Ablation A1 :
- `True` (default, student behavior) : MAE 0.0142
- `False` (2nd-order trains during Grammar-A) : MAE 0.0246

Donc figer le 2nd-order pendant Grammar-A est **un choix sciemment
meilleur**, pas un bug. Le 2nd-order learned pendant pretrain est
mieux que ce qu'il pourrait apprendre en 12 epochs de plus.

### D-agl-epochs-pretrain (D.28.i resolved)

Paper Table 10 dit `n_epochs_pretrain=60`. Student code passe `30`
literal. Sprint-08 D.28.i Ablation A3 :
- Paper-literal 60 → MAE 0.0206
- Student-actual 30 → MAE 0.0142

Donc **student wins**. Paper Table 10 est inconsistant avec le code
qui a produit les Tables 5b/5c. Notre default = 30, override
`-o train.n_epochs_pretrain=60` pour reproduce paper-literal.

## 3. Contraintes d'ingénierie

### `_run_training_loop` extraction

Pattern : éviter duplication training loop entre `AGLTrainer.training`
(single cell) et `AGLNetworkPool.train_range` (N cells).

Module-level fonction takes tous les networks + optimizers + cascade
params comme args explicit. Caller construit ces args.

Légèrement verbeux mais explicite (pas de magie via closure). Test
coverage propre.

### Triple try/except pour `torch_optimizer`

```python
try:
    import torch_optimizer
    _OPTIMIZERS["RANGERVA"] = torch_optimizer.RangerVA
except ImportError:
    log.warning("torch_optimizer not available; RangerVA fallback to ADAMAX")
```

`uv sync --extra agl` installe `torch-optimizer`. Sans cet extra, on
fallback ADAMAX avec warning. Permet `uv sync --extra blindsight`
seul de marcher en mode dégradé.

### `_evaluate_single_cell` helper

Factorisé entre `evaluate` (1 cell) et `evaluate_pool` (N cells).
Génère batch concat Grammar-A + Grammar-B, run cascade forward,
compute metrics (precision_1st, wager_accuracy, precision_2nd,
recall_2nd, f1_2nd).

### `_aggregate_pool_metrics` tiers

High = `[0:num_networks//2]`, Low = `[num_networks//2:]`, Overall =
all. Pour chaque tier : mean + std de chaque key. NaN drop pour les
cells sans 2nd-order keys.

### `pretrain` vs `training` LR différentes

`cfg.optimizer.lr_first_order` (pretrain) ≠
`cfg.optimizer.lr_training_first_order` (training phase). Idem
second-order. Student `initialize_global` distingue les 2.

## 4. Deviations / dettes présentes

### 🚨 D-agl-training-missing (D.28.b-d resolved)

Pre-Sprint-08, AGL trainer n'avait que `pre_train`. Le 3-phase
protocole entier (replicate + training + test) manquait. Sprint-08
D.28 a tout porté → RG-003 closed.

### D-agl-optimizer (D.28.a resolved)

RangerVA preferred (paper T.10), ADAMAX fallback avec warning si
dep missing. Ablation A4 confirms RangerVA essential :
- RangerVA → MAE 0.0142
- ADAMAX → MAE 0.0381

### D-agl-epochs-pretrain (D.28.i resolved)

Voir §2 — 30 wins per ablation A3.

### D-agl-training-meta-override (D.28.h validated)

`train_meta_frozen_in_training=True` student behavior intentional, A1
ablation confirms.

### D-agl-wager-hidden (D.28.a resolved)

Pasquali hidden restored (`hidden_dim=48` pour AGL). Sub-bug
identical au Blindsight D.25 H10. Ablation A5 confirme trade-off
mais on garde pour alignement archi paper.

### D-agl-sched-step/gamma, D-agl-temperature

Tous audités D.28.a, alignés paper.

### Pas de DETTE active

Le module est gros (819 L) mais bien factorisé via les helpers.

## 5. Questions ouvertes

- **Q1 :** Le reset first-order est conceptuellement crucial mais
  implementation-wise fragile. Si on oublie `.deepcopy(state_dict())`
  dans build, on a un bug silencieux. Test parity bit-exact obligatoire.

- **Q2 :** Le `train_meta_frozen` ablation A1 montre que figer le
  2nd-order pendant Grammar-A est mieux. C'est conceptuellement
  étrange — on aurait pensé que le 2nd-order doit s'adapter au new
  domain. Mais empiriquement non. Question pour la science.

- **Q3 :** Pourquoi 20 cells × (12 + 3) epochs et pas, e.g., 10 cells
  × (24 + 6) epochs ? Compute budget identique (300 cell-epochs).
  Ablation possible mais pas faite.

- **Q4 :** *(à toi)* — qu'est-ce qui te paraît surprenant dans le
  3-phase protocole ?

## 6. Pour le rewrite (Sprint 11+)

### À garder
- 3-phase protocole complet (pretrain, training tiers, test)
- Reference reset first-order après pretrain
- `train_meta_frozen=True` default
- RangerVA preferred + ADAMAX fallback
- `_run_training_loop` extracted helper
- AGLSetting 6-cell schema + legacy 2x2 alias

### À améliorer
- **Documenter le reset first-order** plus explicitement (docstring +
  comment near `load_state_dict`)
- **Logging structuré per-tier** : actuellement log.info text, on
  pourrait émettre JSON records
- **Per-tier training override** : currently `n_epochs_training_high`
  et `n_epochs_training_low` séparés en config. OK.
- **Évaluation streaming** : pour 500 seeds × 6 settings × 4 tiers,
  on pourrait stream les metrics dans un duckdb sans tout charger en
  RAM

### Tests à écrire (avant rewrite)
- **Parity bit-exact `pre_train` vs reference** : seed fixé, mêmes
  losses
- **Parity `training` phase** : 1 cell trainé 12 epochs Grammar-A
- **Parity `evaluate_pool`** : sur un pool 20-cell, metrics par tier
  matches reference
- **Test reset** : `pre_train()` puis check `first_order.state_dict()
  == initial`
- **Test `train_meta_frozen=False`** : losses_2 ≠ 0, weights bougent
- **Test fallback ADAMAX** : ImportError torch_optimizer → warning +
  fallback OK

### Opportunités perf
- **Paralléliser pool training** : 20 cells indépendantes →
  `joblib.Parallel(n_jobs=10)` = ~10× speedup local
- **Paralléliser eval** : pareil
- **Cache target_second computation** : actuellement recalculé chaque
  epoch. Mais target change car h2 change. Pas cacheable.
- **Vectoriser per-cell precision computation** : boucle row+chunk
  dans `_run_training_loop` est lente. Vectorisation = ~10× sur cette
  ligne.

### Connexion
- Dépend de : `core.{cascade, second_order, losses}`, `networks.first_order_mlp`,
  `utils.config`, `domains.agl.data`, `domains.agl.pool`
- Appelé par : `domains/agl/cli.py`

## Méta — pour Claude chat

```text
Je travaille sur le protocole AGL du paper MAPS (Vargas et al. 2025)
qui implémente le paradigme implicit vs explicit knowledge de Dienes
(1997). Le mécanisme clé est :

1. Pretrain 1 réseau sur random words (1st-order learns reconstruction,
   2nd-order learns to predict 1st-order success)
2. RESET 1st-order to initial weights, KEEP 2nd-order trained
3. Replicate en 20 cells
4. Train 10 cells 12 epochs Grammar-A (High Awareness), 10 cells 3 epochs
   (Low Awareness)
5. Test : same Grammar-A+B batch on all cells, observe precision +
   wager separation

Questions :
1. Pourquoi le RESET 1st-order après pretrain ? Pourquoi pas garder
   l'apprentissage cumulatif ?
2. Le 2nd-order figé pendant Grammar-A training (`meta_frozen=True`)
   donne MAE 0.014 vs 0.025 si on l'entraîne aussi. Pourquoi figer
   améliore ?
3. Y a-t-il dans la littérature implicit learning des explorations du
   trade-off "1st-order training time × 2nd-order frozen state" qui
   produiraient des effets similaires ?
```
