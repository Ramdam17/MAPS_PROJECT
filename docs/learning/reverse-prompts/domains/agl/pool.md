# Reverse-prompt — `src/maps/domains/agl/pool.py`

**Module path actuel (sur main) :** `src/maps/experiments/agl/pool.py`
**Taille :** ~225 lignes.
**Paper :** §A.2, Table 10 (epochs 12 / 3 pour High / Low awareness).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `pool.py` qui implémente la **réplication
> 20-networks** post-pretrain du protocole AGL (paper §A.2). Porte
> `AGL/AGL_TMLR.py:create_networks` L814-892 du student.
>
> **Protocole paper §A.2** :
> 1. Pretrain 1 réseau sur random-grammar words (`AGLTrainer.pre_train`)
> 2. Reset le first-order à ses **initial weights** (paper L751)
> 3. Replicate en **20 cells indépendantes** chacune avec :
>    - First-order = reset initial weights (deep-copied)
>    - Second-order = post-pretrain weights (deep-copied)
>    - Fresh per-cell optimizers (training-phase LR)
>    - Fresh per-cell schedulers
> 4. Split en 2 tiers :
>    - **High Awareness** = cells [0:10] → train 12 epochs Grammar-A
>    - **Low Awareness** = cells [10:20] → train 3 epochs Grammar-A
> 5. Evaluate les 20 cells, agrège par tier (mean ± std)
>
> **Structures de données** :
> ```python
> @dataclass
> class NetworkCell:
>     first_order: FirstOrderMLP
>     second_order: SecondOrderNetwork
>     optim_1: Optimizer
>     optim_2: Optimizer
>     sched_1: StepLR
>     sched_2: StepLR
>
> class AGLNetworkPool:
>     def __init__(self, trainer: AGLTrainer, num_networks: int)
>     def __len__(self) -> int
>     def train_range(self, start, end, n_epochs) -> dict[str, np.ndarray]
> ```
>
> **`train_range(start, end, n_epochs)`** délègue à
> `_run_training_loop` (helper module-level dans `trainer.py`) pour
> chaque cell dans `[start, end)`. Évite duplication code (~150 lignes
> partagées avec `AGLTrainer.training`).
>
> **Deep-copy via `copy.deepcopy(state_dict())`** — en mémoire, pas de
> disk round-trip (le student L826-890 utilise `torch.save` to disk
> avec `PATH` global). Notre version est plus propre et plus rapide.

## 2. Contraintes scientifiques

### Pourquoi 20 cells ?

Le paper §A.2 dit explicitement 20 (Table 10). C'est une moyenne sur
20 instances pour réduire la variance de l'estimation High/Low. Pas
de raison théorique forte ; juste un compromis variance vs compute.

### High vs Low Awareness

Le 1st-order vide a un budget d'apprentissage très différent (12 vs 3
epochs Grammar-A). Au moment du test, les 2 tiers ont :
- Le **même** 2nd-order (issu du pretrain partagé)
- Des first-orders **différemment entraînés** sur Grammar-A

Le wager apprend dans le pretrain à prédire "ce 1st-order a-t-il
réussi ?". À l'inférence sur Grammar-A :
- High Awareness : 1st-order bien entraîné → succès souvent → wager
  high (consciousness explicite)
- Low Awareness : 1st-order peu entraîné → succès rare → wager low
  (consciousness implicite)

C'est le mécanisme **MAPS de la dissociation consciousness/precision**
en AGL. Très élégant.

## 3. Contraintes d'ingénierie

### Memory cost

20 cells × (first_order + second_order) en RAM. Pour AGL c'est ~100 MB
total (réseaux petits). OK même en dev local.

Si on scale-up (plus de cells, plus gros networks), ça devient un
constraint. Pour MARL ce serait impossible.

### Fresh per-cell optimizers

Le student fait `create_networks` qui rebuild optimizers from scratch
pour chaque cell. Important : sinon les momentum/adaptive states du
pretrain optimizer fuiraient dans les training-phase optimizers.

Notre port mirror ça : `_build_optimizer(opt_cfg.name,
fo.parameters(), lr=lr_training_first_order)` pour chaque cell.

### `bits_per_letter` unused warning

Le code `_ = bits_per_letter` est un placeholder qui signale "ce param
serait utile mais le passe-through au constructor de FirstOrderMLP
est fait via `make_chunked_sigmoid(bits_per_letter)` ailleurs".
Cosmétique.

### `_run_training_loop` shared helper

Pour éviter la duplication entre `AGLTrainer.training` (un cell) et
`AGLNetworkPool.train_range` (N cells), le training loop est extrait
en module-level fonction dans `trainer.py`. Pool importe.

Pattern d'architecture propre. Évite la cascade de copy-paste.

## 4. Deviations / dettes présentes

### D-agl-training-missing (D.28.b-d resolved)

Avant Sprint-08 D.28, ce module n'existait pas. Le port AGL faisait
seulement le pretrain phase, manquait les 2 phases suivantes. Sprint
08 D.28 a porté `create_networks` + `training` + `testing` → ce
module.

Resolution complète : RG-003 fermé, MAE 0.0142 sur 12 metrics paper
Table 5b/5c.

### Pas de DETTE active

Module propre, bien typé, bien testé.

## 5. Questions ouvertes

- **Q1 :** Pourquoi en-memory deep-copy vs torch.save sur disque (student) ?
  Notre choix est plus rapide et plus propre. Mais le student le faisait
  pour pouvoir restart un training depuis un cell donné. On perd ça.
  Trade-off acceptable.

- **Q2 :** Cells size = 10 / 10 dur dans la mid. Si `num_networks=22`,
  on aurait 11 / 11. Si `num_networks=21`, on aurait 10 / 11 (split //
  2). Édge case OK avec student behavior ?

- **Q3 :** Pourrait-on parallélisre `train_range` ? Les cells sont
  indépendantes. Avec `joblib.Parallel(n_jobs=10)` on aurait 10× speedup
  local. Trade-off : seed determinism + GPU memory.

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- 20-cell replication via deep-copy
- High/Low tier split via `train_range(start, end, n_epochs)`
- Fresh per-cell optimizers
- Shared `_run_training_loop` helper

### À améliorer
- **Paralléliser `train_range`** : `joblib.Parallel(n_jobs=N)` sur
  les cells. Gain massif (4-8×). Requires seeding per cell.
- **Save / restore pool checkpoint** : pour pouvoir restart un
  pool training depuis un cell donné (recover du student behavior)
- **Tier split paramétrable** : currently `num_networks // 2`.
  Permettre `split_idx` config-driven (utile si on veut 5/15 split).

### Tests à écrire
- 20-cell pool a bien 20 cells après init
- Each cell independent (modifying cell[0].first_order ne touche pas
  cell[1])
- `train_range(0, 10, 12)` ne modifie pas cells [10:20]
- Per-cell precision varies (cells diffèrent post-training)
- Aggregate tiers : high vs low diffèrent (sinon le mécanisme ne
  marche pas)

### Connexion
- Utilise `AGLTrainer.cfg` + `.device`
- Utilise `_run_training_loop` from `trainer.py`
- Utilise `FirstOrderMLP, SecondOrderNetwork`, optimizers
- Appelé par `domains/agl/cli.py`
