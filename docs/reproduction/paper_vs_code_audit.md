# Paper vs code — cross audit

**Source documents (populated in Phase B) :**

- **Paper :** `pdf/MAPS_TMLR_Journal_Submission.pdf` + extractions verbatim dans
  `docs/reproduction/paper_{tables,equations,targets}_extracted.md`.
- **Student monoliths :** `external/paper_reference/{sarl,sarl_cl,agl,blindsight}_*.py` +
  shell launchers récupérés de git history (`git show 8c38e4f^:SARL/*.sh`).
- **Current ports + configs :** `src/maps/experiments/*`, `config/training/*.yaml`,
  `config/maps.yaml`, et quelques constantes hardcodées dans `trainer.py`.

Pour chaque hyperparam / choix architectural : 5 colonnes (Paper, Student monolith, Our port+config,
∆ verdict, Deviation ID vers `deviations.md`).

**Conventions verdict :**
- ✅ match (within float precision)
- ⚠️ code-vs-code divergence mineure ou paper silent — à documenter, non bloquant
- ❌ port-vs-paper divergence (paper explicit, port disagrees) → fix dans Phase D
- 🆘 paper-vs-student divergence (**student code ne reproduit pas la Table 11 du papier** — signal
  le plus grave, car même tourner le code student comme publié ne reproduirait pas les z-scores
  papier)

Phase B du plan `docs/plans/plan-20260419-review-and-reproduce.md` remplit ce doc sous-phase par
sous-phase (B.7 SARL, B.8 SARL+CL, B.9 Blindsight, B.10 AGL, B.11 MARL). Phase B.12 consolide
l'ensemble dans `deviations.md`.

---

## SARL — MinAtar DQN (B.7)

**Sources comparées :**

- Paper Table 11 (p. 30) + eqs. 1-14 (pp. 6-8), extractions dans `paper_{tables,equations}_extracted.md`.
- Student monolith `external/paper_reference/sarl_maps.py` (2 721 lignes). Plus shell launcher
  `SARL/SARL_Training_Standard.sh` (restauré via `git show 8c38e4f^:`) qui fournit `-ema 25`.
- Our port : `src/maps/experiments/sarl/` (1 555 lignes, 7 fichiers), `config/training/sarl.yaml`,
  + une constante hardcodée `GAMMA = 0.99` dans `trainer.py:54`.

### 1. Hyperparameters — Table 11 vs student vs port

| # | Param                              | Paper T.11         | Student `sarl_maps.py`                              | Our port + config                          | ∆    | Dev ID                    |
|---|------------------------------------|-------------------:|----------------------------------------------------:|-------------------------------------------:|:----:|:-------------------------|
| 1 | Batch size                         | 128                | `BATCH_SIZE = 128` (L92)                            | `training.batch_size = 128`                | ✅   | —                         |
| 2 | Replay buffer size                 | 100,000            | `REPLAY_BUFFER_SIZE = 100000` (L93)                 | `training.replay_buffer_size = 100_000`    | ✅   | —                         |
| 3 | Target network update freq         | **1,000**          | **`TARGET_NETWORK_UPDATE_FREQ = 100`** (L1188,1193, set inside `dqn()`) | `training.target_update_freq = 1_000`      | 🆘   | D-sarl-target-update      |
| 4 | Training frequency                 | 1                  | `TRAINING_FREQ = 1` (L94)                           | `training.training_freq = 1`               | ✅   | —                         |
| 5 | Number of frames                   | **500,000** (text: 1M) | `args.steps` CLI (no default)                   | `training.num_frames = 5_000_000`          | ❌   | D-sarl-num-frames         |
| 6 | First N frames (ε-anneal)          | 100,000            | `FIRST_N_FRAMES = 100000` (L95)                     | (hardcoded `rollout.anneal_epsilon`)       | ✅   | —                         |
| 7 | Replay start size                  | 5,000              | `REPLAY_START_SIZE = 5000` (L96)                    | `training.replay_start_size = 5_000`       | ✅   | —                         |
| 8 | End epsilon                        | 0.1                | `END_EPSILON = 0.1` (L97)                           | (hardcoded `rollout.anneal_epsilon`)       | ✅   | —                         |
| 9 | lr first-order (step_size)         | 0.0003             | `step_size1 = 0.0003` (L98)                         | `optimizer.lr_first_order = 0.0003`        | ✅   | —                         |
|10 | lr second-order                    | **0.0002**         | **`step_size2 = 0.00005`** (L99)                    | `optimizer.lr_second_order = 0.00005`      | 🆘+❌ | D-sarl-lr-2nd             |
|11 | Adam β₁ (Gradient momentum)        | **0.95**           | `GRAD_MOMENTUM = 0.95` (L101) **mais non passé à Adam** | PyTorch default 0.9                     | 🆘+❌ | D-sarl-adam-beta1         |
|12 | Adam β₂ (Squared gradient momentum)| **0.95**           | `SQUARED_GRAD_MOMENTUM = 0.95` (L102) **mais non passé à Adam** | PyTorch default 0.999              | 🆘+❌ | D-sarl-adam-beta2         |
|13 | Adam eps (Minimum squared gradient)| 0.01               | `MIN_SQUARED_GRAD = 0.01` (L103), passed via `eps=...` | `optimizer.eps = 0.01`                   | ✅   | —                         |
|14 | Gamma (DQN discount)               | **0.999**          | **`GAMMA = 0.99`** (L104)                           | **`trainer.py:54 GAMMA = 0.99`** hardcoded | 🆘+❌ | D-sarl-gamma              |
|15 | Scheduler step size                | **1**              | **`StepLR(..., step_size=1000, ...)`** (L1314,1319) | `scheduler.step_size = 1_000`              | 🆘+❌ | D-sarl-sched-step         |
|16 | Scheduler gamma                    | (absent table)     | `scheduler_step = 0.999` (L107) → gamma de StepLR   | `scheduler.gamma = 0.999`                  | ⚠️   | D-sarl-sched-gamma (info) |
|17 | Epsilon (exploration start)        | 1.0                | `EPSILON = 1.0` (L105)                              | (hardcoded `rollout.anneal_epsilon`)       | ✅   | —                         |
|18 | **Alpha (EMA wagering)**           | **0.45**           | `-ema 25` shell → `/100` = **0.25** (`target_wager` L516) | `config.alpha = 1.0` → `/100` = **0.01** | 🆘🆘 | D-sarl-alpha-ema          |
|19 | Cascade iterations                 | 50                 | `cascade_1=50, cascade_2=50` setting 6 block        | `setting_to_config` match                  | ✅   | —                         |
|20 | Optimizer                          | Adam               | `optim.Adam(...)` (L1313, 1318)                     | `optimizer.name = "Adam"`                  | ✅   | —                         |

**20 hyperparam rows. Divergences identifiées : 8 (dont 5 🆘 paper-vs-student).**

**Note critique sur les Adam betas** (rows 11-12) : le student définit `GRAD_MOMENTUM = 0.95` et
`SQUARED_GRAD_MOMENTUM = 0.95` au niveau module (comme si c'était Table 11), **mais n'utilise jamais
ces constantes** — l'appel `optim.Adam(params, lr=STEP_SIZE, eps=MIN_SQUARED_GRAD)` ligne 1313
n'a pas de `betas=...` argument. **PyTorch default** `betas=(0.9, 0.999)` s'applique. Donc le code
student tourne avec des betas **différents** de ceux que le papier annonce. C'est ou bien une erreur
du student qui a oublié de passer le paramètre, ou bien les valeurs 0.95/0.95 de Table 11 sont
elles-mêmes fausses. Dans les deux cas, divergence à flagger.

### 2. Architectural choices

| # | Item                             | Paper eq. / text                              | Student code                                                           | Our port                             | ∆    | Dev ID                     |
|---|----------------------------------|-----------------------------------------------|------------------------------------------------------------------------|--------------------------------------|:----:|:--------------------------|
| A | Conv kernel / filters            | paper silent in body ; fig. 3                 | `nn.Conv2d(in, 16, kernel_size=3, stride=1)`                           | Same (`model.py:73`)                  | ✅   | —                          |
| B | Hidden dim (fc_hidden)           | paper silent                                  | `nn.Linear(1024, 128)` (L1024 → L138)                                  | Same                                  | ✅   | —                          |
| C | Tied-weight reconstruction bias  | eq. 12: `ReLU(W^T H^{(i)} + b_recon)`         | `F.relu(F.linear(Hidden, fc_hidden.weight.t()))` — **no bias term**    | Same as student (no bias)             | 🆘+❌ | D-sarl-recon-bias          |
| D | Dropout position                 | eq. 2 : on `C_t` **before** cascade           | `self.dropout(comparison_matrix)` inside cascade loop → called 50× / update for setting 6 | Same as student (inside loop)         | 🆘+⚠️ | D-sarl-dropout-position    |
| E | Wagering head output             | eq. 3 : 2 raw logits                          | `nn.Linear(1024, 2)`, init `uniform_(0, 0.1)`                          | Same                                  | ✅   | —                          |
| F | Dropout rate                     | paper silent                                  | `p=0.1`                                                                | Same                                  | ⚠️   | D-sarl-dropout-rate (info) |
| G | Wagering BCE shape               | eq. 5 : scalar y                              | `F.binary_cross_entropy_with_logits(wager[B,2], targets[B,2])`         | Same                                  | ⚠️   | D-sarl-bce-shape           |
| H | **Main-task loss**               | **eq. 4 : SimCLR contrastive**                | **`cae_loss` (Rifai 2011 CAE, different from eq. 4)**                  | Same (CAE)                            | 🆘+❌ | D-002 (already open)       |
| I | Backward order (meta branch)     | paper silent                                  | `loss_second.backward(retain_graph=True); optimizer2.step(); loss.backward(); optimizer.step()` — **load-bearing** | Same                                  | ⚠️   | D-sarl-backward-order      |
| J | Replay buffer type               | paper silent                                  | Python `list` + `random.sample`                                        | Same                                  | ✅ (match student) | —                |

**10 architectural rows. Divergences identifiées : 3 sérieuses (recon-bias, dropout-position, CAE-vs-SimCLR) + 3 ⚠️ informatives.**

### 3. Seed count

| Source                                                  | Seeds |
|---------------------------------------------------------|:-----:|
| Paper main text p. 13 + Table 6 caption                  | **3** |
| Our `docs/reproduction/experiment_matrix.md` (pre-B.13)  | 10 ❌ |
| Sprint-07 plan "décisions verrouillées 2026-04-18"       | 10 ❌ |

**D-sarl-seeds** : correction à N=3 dans Phase B.13.

### 4. Setting factorial

| # | Setting         | Paper (fig. 6 + Table 6)                       | Our port (`setting_to_config`)              | ∆   | Dev ID               |
|---|-----------------|------------------------------------------------|---------------------------------------------|:---:|:--------------------|
| 1 | 1 Baseline      | No cascade, no meta                            | (meta=F, cascade_1=1, cascade_2=1)          | ✅  | —                    |
| 2 | 2               | Cascade on 1st net only                        | (meta=F, cascade_1=50, cascade_2=1)         | ✅  | —                    |
| 3 | 3               | Meta on, no cascade                            | (meta=T, cascade_1=1, cascade_2=1)          | ✅  | —                    |
| 4 | 4 MAPS          | Meta + cascade on 1st net                      | (meta=T, cascade_1=50, cascade_2=1)         | ✅  | —                    |
| 5 | 5               | Meta + cascade on 2nd net                      | (meta=T, cascade_1=1, cascade_2=50)         | ✅  | —                    |
| 6 | 6 full MAPS     | Meta + cascade on both                         | (meta=T, cascade_1=50, cascade_2=50)        | ✅  | —                    |
| 7 | **7 (ACB)**     | **Actor-Critic Baseline, Young & Tian 2019, λ=0.8** | **NOT IN PORT** (`AC_lambda.py` in `external/MinAtar/examples/` + restored to `external/paper_reference/sarl_ac_lambda.py`) | ❌  | D-sarl-setting-7     |

### 5. Summary of SARL divergences

**Deviation IDs à enregistrer dans `deviations.md` (Phase B.12).** Ordre par sévérité décroissante :

| # | Deviation ID             | Paper     | Student     | Port              | Verdict | Action plan                 |
|---|--------------------------|----------:|------------:|------------------:|:-------:|:----------------------------|
| 1 | D-sarl-alpha-ema         | 0.45      | 0.25        | 0.01              | 🆘🆘   | D.2 — config + docs         |
| 2 | D-sarl-gamma             | 0.999     | 0.99        | 0.99 (hardcoded)  | 🆘+❌   | D.7 — gamma config-driven   |
| 3 | D-sarl-lr-2nd            | 2e-4      | 5e-5        | 5e-5              | 🆘+❌   | D.9 — align lr_2            |
| 4 | D-sarl-target-update     | 1000      | 100 (L1188) | 1000              | 🆘     | **port déjà correct**       |
| 5 | D-sarl-sched-step        | 1         | 1000        | 1000              | 🆘+❌   | D.9 — scheduler step = 1    |
| 6 | D-sarl-adam-beta1        | 0.95      | 0.9 (unused const) | 0.9 (default) | 🆘+❌   | D.9 — Adam betas config     |
| 7 | D-sarl-adam-beta2        | 0.95      | 0.999 (unused const) | 0.999 (default) | 🆘+❌   | D.9 — Adam betas config     |
| 8 | D-sarl-num-frames        | 500k/1M   | CLI         | 5M                | ❌      | D.12 — align on paper       |
| 9 | D-002 (contrastive-vs-CAE) | SimCLR (eq. 4) | CAE     | CAE               | ❌+🆘   | C.7-C.9 — policy decision   |
|10 | D-sarl-recon-bias        | `+b_recon`| no bias     | no bias           | ❌+🆘   | D.3 — add bias              |
|11 | D-sarl-dropout-position  | before cascade | inside loop | inside loop   | ❌+⚠️   | D.4 — move dropout outside  |
|12 | D-sarl-setting-7         | ACB present | AC_lambda.py | not ported      | ❌      | E.1-E.5 — port ACB          |
|13 | D-sarl-seeds             | 3         | n/a         | 10 (matrix)       | ❌      | B.13 — correct matrix       |
|14 | D-sarl-bce-shape         | scalar y  | 2-unit      | 2-unit            | ⚠️     | keep, doc note              |
|15 | D-sarl-dropout-rate      | silent    | p=0.1       | p=0.1             | ⚠️     | keep, doc note              |
|16 | D-sarl-sched-gamma       | silent    | 0.999       | 0.999             | ⚠️     | keep, doc note              |
|17 | D-sarl-backward-order    | silent    | specific    | specific          | ⚠️     | keep, load-bearing          |

**Finding structurel :** 7 des 17 divergences sont **🆘 paper-vs-student** — le code student tel que
publié n'a **jamais** pu produire les chiffres de Table 11. Les z-scores de Table 6 viennent d'une
version du code qui n'est pas celle vendored ici. Conséquences pour Phase F :

- Aligner notre port sur la paper Table 11 **peut** diverger de ce que le student a publié en Table 6.
- Documenter ça comme une **limitation connue** dans le rapport final (Phase G) : *"Paper Table 11
  hyperparameters were applied faithfully; if the paper's own Table 6 z-scores derive from a
  different hyperparameter set, our numbers will differ."*

**Next** : Phase B.8 audit SARL+CL (section suivante de ce même doc).

---

## SARL+CL — MinAtar DQN + continual learning (B.8)

**Sources comparées :**

- Paper Table 11 (p. 30) — mêmes 20 rows que SARL **plus** 4 rows CL-specific (max_input_channels,
  weight task loss, weight weight regularization loss, weight feature loss). Paper eqs. 15-18
  (p. 9) pour les 3 losses CL. Paper §CL Results p. 17 + §Discussion p. 21 mentionnent un "optimal
  identified" `task=0.4, reg=0.4, feature=0.2` qui **diffère** de Table 11 — **paper disagrees
  with itself**.
- Student monolith `external/paper_reference/sarl_cl_maps.py` (2 580 lignes). CLI defaults
  `--weight1 40 --weight2 40 --weight3 20` → /100 = **(0.4, 0.4, 0.2)** — student suit le **texte
  paper**, pas Table 11.
- Our port : `src/maps/experiments/sarl_cl/` (1 536 lignes, 4 fichiers : `model.py`, `trainer.py`,
  `training_loop.py`, `loss_weighting.py`) + `config/training/sarl_cl.yaml`.

### 1. Hyperparameters — delta-from-SARL

Les 20 rows de la section SARL s'appliquent identiquement à SARL+CL (même monolith `BATCH_SIZE=128`,
`step_size1=0.0003`, `GAMMA=0.99` hardcoded, `scheduler_step=0.999`, Adam sans betas, etc.) —
**toutes les divergences SARL sont héritées par SARL+CL**. Les rows ci-dessous listent uniquement
**ce qui diffère ou s'ajoute** par rapport à SARL.

| #  | Param                                    | Paper T.11        | Paper text (p. 17,21) | Student `sarl_cl_maps.py`                       | Our port + config                   | ∆    | Dev ID                       |
|----|------------------------------------------|------------------:|----------------------:|------------------------------------------------:|------------------------------------:|:----:|:-----------------------------|
| 21 | **Target network update freq (CL)**      | 1,000 (same row as SARL) | —               | **500** (L1121, 1126, scoped inside `dqn()`)    | `training.target_update_freq = 500` | ⚠️   | D-sarl_cl-target-update      |
| 22 | **Max input channels (CL)**              | **10**            | —                     | `MAX_INPUT_CHANNELS = 10` (L90)                 | `cl.max_input_channels = 7` ❌      | ❌   | D-sarl_cl-max-channels       |
| 23 | **weight task loss** (λ_task)            | **0.3**           | 0.4 ("optimal identified") | CLI default `--weight1=40` → **0.4**        | `cl.weight_task = 1.0` ❌ ❌        | 🆘+❌ | D-cl-weights                 |
| 24 | **weight reg loss** (λ_reg)              | **0.6**           | 0.4                   | CLI default `--weight2=40` → **0.4**            | `cl.weight_distillation = 1.0` ❌   | 🆘+❌ | D-cl-weights                 |
| 25 | **weight feature loss** (λ_feature)      | **0.1**           | 0.2                   | CLI default `--weight3=20` → **0.2**            | `cl.weight_feature = 1.0` ❌        | 🆘+❌ | D-cl-weights                 |
| 26 | num_frames (per env in curriculum)       | (inherits SARL row 5 — 500k / 1M) | p. 17: 100,000 | `args.steps` CLI                      | `training.num_frames = 5_000_000` ❌ | ❌   | D-sarl_cl-num-frames         |

**Note poids CL (rows 23-25)** : **triple-inconsistency**.
- Paper Table 11 : `(0.3, 0.6, 0.1)`.
- Paper text p. 17 + p. 21 (discussion CL) : `(0.4, 0.4, 0.2)` décrit comme "optimal identified for
  knowledge retention".
- Student actual runs : `(0.4, 0.4, 0.2)` — suivent le texte, pas la table. Les z-scores de paper
  §CL Results p. 17 ont donc probablement été obtenus avec ces poids, pas ceux de Table 11.
- Our port : `(1.0, 1.0, 1.0)` — unnormalized, ne matche ni Table 11 ni le texte. Somme = 3,
  pas 1 (eq. 18 exige `λ_task + λ_reg + λ_feature = 1`).

**Policy Rémy (paper = source of truth)** : on aligne sur **Table 11** `(0.3, 0.6, 0.1)` par défaut,
et on documente dans `deviations.md` que le texte paper p. 17 et le student ont utilisé
`(0.4, 0.4, 0.2)`. Phase F peut ajouter un run alternatif avec les valeurs "text" pour comparaison.

**Note max_input_channels (row 22)** : port dit 7, paper dit 10. Seaquest a 10 canaux → avec
`max=7` on **tronque** Seaquest → perte d'info silencieuse. Très vraisemblablement un bug
introduit en Sprint-04b CL port. À fix Phase D.20.

**Note num_frames CL (row 26)** : paper §CL p. 17 "trained sequentially for 100,000 steps ... for
each of the 4 environments". Notre config à 5M = tout faux côté CL (cellule unique devrait être
100k × 4 envs = 400k total).

### 2. Architectural choices — CL-specific

| # | Item                                    | Paper                                           | Student code                                              | Our port                                        | ∆    | Dev ID                          |
|---|-----------------------------------------|-------------------------------------------------|-----------------------------------------------------------|-------------------------------------------------|:----:|:-------------------------------|
| K | Teacher network (frozen, prev task)     | §CL p. 9 : *"current task network (main task network) and the previous task network (teacher network)"* | `load_file_path` loads teacher weights ; frozen forward   | `SarlCLQNetwork` + teacher loaded from ckpt     | ✅   | —                                |
| L | 3-loss hybrid                           | eq. 15-18 (L_reg, L_feature, L_task, L_continual) | `train()` computes 3 losses + combined                    | `trainer.sarl_cl_update_step` + `loss_weighting` | ✅   | —                                |
| M | **Running-max normalization** (eq. 15-17 `1/max_t(L(t))`) | eq. 15-17 : chaque loss divisée par son `max_t(L(t))` | Inline in `train()` L ~1300 | `DynamicLossWeighter` class | ⚠️ | D-sarl_cl-lossweight-normalization (info) |
| N | Variable-channel conv adapter           | p. 9 : *"1×1 convolution layer with ReLU to process inputs of different sizes"* | ? (to verify) | `SarlCLQNetwork` with `in_channels=max_input_channels` + zero-padding | ⚠️ | D-sarl_cl-channel-adapter (to verify) |
| O | Backward order (meta + 3-loss branch)   | paper silent                                    | specific order (load-bearing per Sprint-04b docstring)    | Same (`trainer.py` mirrors student)             | ⚠️   | D-sarl_cl-backward-order (info)  |
| P | Curriculum env order                    | p. 9 : *"Breakout, Space Invaders, Seaquest, and Freeway"* | ? (to verify)                        | `scripts/run_sarl_cl.py` + config               | ⚠️   | D-sarl_cl-curriculum-order       |

**6 architectural rows. Divergences identifiées : 0 hard (tout ✅ ou ⚠️ à vérifier Phase D.16-D.19).**

### 3. Seed count

Même convention que SARL — **N = 3 seeds** (paper text p. 17 :
*"we trained sequentially for 100,000 steps (due to computational limitations faced when using
teacher networks) for each of the 4 environments defined in our curriculum. ... Each subplot
represents a different MinAtar game ... N = 3 seeds per setting."*). **D-sarl-seeds** s'applique
identiquement ici (matrix dit 10, paper dit 3, à fix Phase B.13).

### 4. Setting factorial + curriculum

Les 7 settings SARL (1-6 MAPS factorial + 7 ACB) sont **hérités**. CL ajoute un axe curriculum :

| Curriculum stage | Paper (p. 9, 17)      | Our port                                     | ∆   |
|------------------|-----------------------|----------------------------------------------|:---:|
| Stage 1          | Breakout              | (to verify `config/training/sarl_cl.yaml` ou CLI) | ⚠️ |
| Stage 2          | Space Invaders        | idem                                         | ⚠️ |
| Stage 3          | Seaquest              | idem                                         | ⚠️ |
| Stage 4          | Freeway               | idem                                         | ⚠️ |

**Note ACB pour SARL+CL** : Table 11 papier ne mentionne pas ACB pour CL. Paper §CL Results p. 17
ne compare pas à ACB — focus retention/forgetting, pas z-scores absolus. Donc Setting 7 peut
légitimement être absent du port CL. À confirmer Phase D.19.

### 5. Summary of SARL+CL divergences

**Divergences propres à CL (en plus des 17 SARL héritées) :**

| # | Deviation ID                      | Paper T.11    | Paper text     | Student CLI | Port             | Verdict | Action                                |
|---|-----------------------------------|--------------:|---------------:|------------:|-----------------:|:-------:|:--------------------------------------|
| A | D-cl-weights (×3 params)          | (0.3,0.6,0.1) | (0.4,0.4,0.2)  | (0.4,0.4,0.2) | (1.0,1.0,1.0) ❌ | 🆘+❌   | D.20 — align sur Table 11 (policy)    |
| B | D-sarl_cl-max-channels            | 10            | —              | 10          | **7** ❌          | ❌      | D.20 — fix config to 10               |
| C | D-sarl_cl-num-frames              | (inherits)    | 100,000/env    | CLI         | 5M ❌             | ❌      | D.20 — align to 100k per env          |
| D | D-sarl_cl-target-update           | 1,000         | —              | **500**     | 500              | ⚠️      | port matches student ; paper silent   |
| E | D-sarl_cl-lossweight-normalization| eq. 15-17 `1/max_t` | —         | inline      | `DynamicLossWeighter` | ⚠️ | keep, doc note                        |
| F | D-sarl_cl-channel-adapter         | 1×1 conv ReLU | —              | —           | zero-padding + max | ⚠️   | D.16-D.17 — verify impl               |
| G | D-sarl_cl-curriculum-order        | Br→SI→Sq→Fr   | —              | —           | to verify        | ⚠️      | D.19 — confirm                        |
| H | D-sarl_cl-backward-order          | silent        | —              | specific    | same             | ⚠️      | keep, load-bearing                    |

Plus **17 déviations SARL héritées** de B.7 qui s'appliquent toutes (alpha-ema, gamma, lr-2nd,
adam-beta1/2, sched-step, num-frames, recon-bias, dropout-position, D-002 contrastive vs CAE,
etc.).

**Total SARL+CL : 8 CL-specific + 17 inherited from SARL = 25 deviation IDs.**

**Finding critique CL-specific :**
- **D-cl-weights** : seule divergence **🆘+❌** propre à CL. Triple-inconsistency (table paper /
  text paper / student / port tous différents). Policy **paper Table 11** `(0.3, 0.6, 0.1)` par
  défaut.
- **D-sarl_cl-max-channels** : bug simple (config 7 vs paper 10). Seaquest tronqué.
- Autres divergences CL-specific sont informationnelles ou dépendent de vérifications ultérieures
  Phase D.16-D.22.

**Next** : Phase B.9 audit Blindsight (section suivante).

---

## Blindsight (B.9)

**Sources comparées :**

- Paper Table 9 (p. 28) + eqs. 1-6 (Know-Thyself, p. 6). Table 5a cible z-scores (p. 13,
  suprathreshold condition).
- Student monolith `external/paper_reference/blindsight_tmlr.py` (2 419 lignes). **Pas de
  constantes module-level** : `main()` (L2164) définit une **grid search**
  `hidden_sizes = [30, 40, 50, 60, 100]`, `step_sizes = [12, 25, 50]`, `gammas = [0.98, 0.99]`,
  `optimizer = ['ADAMAX']`, `cascade_mode = 0.02`, `seeds = 5` (basic) / `seeds_violin = 10`
  (stat plots). Table 9 paper reporte les **valeurs sélectionnées** après la grid search
  (hidden=60, step=25, gamma=0.98).
- Our port : `src/maps/experiments/blindsight/` (~600 LOC — `trainer.py` 442 L, `data.py` 161 L,
  `__init__.py` 29 L) + `config/training/blindsight.yaml`.

### 1. Hyperparameters — Table 9 vs student grid vs port

| #  | Param                              | Paper T.9       | Student `blindsight_tmlr.py`                                | Our port + config                             | ∆    | Dev ID                          |
|----|------------------------------------|----------------:|------------------------------------------------------------:|----------------------------------------------:|:----:|:--------------------------------|
| B1 | Input size                         | 100             | 100 (paper §A.1 dataset, not a CLI param)                   | `first_order.input_dim = 100`                 | ✅   | —                               |
| B2 | Output size                        | 100             | 100 (autoencoder reconstructs input)                        | (implicit: same as input_dim)                 | ✅   | —                               |
| B3 | **Hidden size**                    | **60**          | grid `[30, 40, 50, 60, 100]` → Table 9 selects 60           | **`first_order.hidden_dim = 100`** ❌         | 🆘+❌ | D-blindsight-hidden-dim         |
| B4 | lr first order                     | 0.5             | `learning_rate_1` (from train() signature, passed via main) | `optimizer.lr_first_order = 0.5`              | ✅   | —                               |
| B5 | lr second order                    | 0.1             | `learning_rate_2`                                           | `optimizer.lr_second_order = 0.1`             | ✅   | —                               |
| B6 | Temperature                        | 1.0             | (implicit — softmax without custom T in autoencoder forward) | (config silent)                             | ⚠️   | D-blindsight-temperature        |
| B7 | Step size (scheduler)              | 25              | grid `[12, 25, 50]`                                          | `scheduler.step_size = 25`                    | ✅   | —                               |
| B8 | Gamma (scheduler)                  | 0.98            | grid `[0.98, 0.99]`                                          | `scheduler.gamma = 0.98`                      | ✅   | —                               |
| B9 | Epochs                             | 200             | grid search trains configs to **some** number (to verify)    | `train.n_epochs = 200`                        | ⚠️   | D-blindsight-epochs (verify)    |
| B10| Optimizer                          | Adamax          | `optim.Adamax(...)` L454 (5 optimizer options available)    | `optimizer.name = "ADAMAX"`                   | ✅   | —                               |
| B11| Cascade iterations                 | 50              | `cascade_mode = 0.02` → `int(1.0/0.02) = 50`                | (`maps.yaml` `cascade_iterations = 50`)       | ✅   | —                               |

**11 hyperparam rows. Divergences majeures : 1 (hidden_dim).**

**Note hidden_dim critique (row B3)** : paper Table 9 explicitly says 60 — résultat de la grid
search du student (hidden_sizes=[30, 40, 50, 60, 100]). Notre port met **100**. 100 est la taille
de l'input ; le commentaire du config dit *"Blindsight reuses input dim as hidden"*. **Mais ce
n'est PAS ce que fait le papier** : le papier dit 60 (goulot de l'autoencoder, pas identité).
Avec hidden=100, notre autoencoder **n'a plus de bottleneck** — la reconstruction devient
triviale, la comparison matrix devient ≈ 0, le wager n'a plus rien à lire. **C'est une cause
structurelle plausible de RG-002.** À fix Phase D.25.

**Note optimizer (row B10)** : student a **5 optimizer options** implémentées dans `pre_training()`
(Adam, AdamW, RMSprop, Adamax, RangerVA — lignes 434-459). Table 9 selectionne Adamax. Our port
aussi. Toutefois le student run réel peut avoir testé plusieurs — à confirmer avant conclure port
correct.

### 2. Architectural choices

| # | Item                                   | Paper (§2.1 + eq.1-6)                             | Student code                                                            | Our port                                                  | ∆    | Dev ID                          |
|---|----------------------------------------|---------------------------------------------------|-------------------------------------------------------------------------|-----------------------------------------------------------|:----:|:-------------------------------|
| Q | Architecture                           | Autoencoder + comparator matrix + 2-wager head   | `FirstOrderNetwork` (AE) + `SecondOrderNetwork` (comparator + wager)    | `FirstOrderMLP` (AE) + `SecondOrderNetwork` (per `maps.yaml`) | ✅   | —                                |
| R | Dropout rate (wager head)              | paper silent                                      | `nn.Dropout(0.1)` (L160) in wager + `nn.Dropout(0.5)` (L222) elsewhere  | per `maps.yaml` (dropout const)                           | ⚠️   | D-blindsight-dropout-rate       |
| S | Cascade position                       | eq. 6 on activations                              | applied in `pre_training` forward (inside loops L536-567)               | `cascade_update` in `FirstOrderMLP.forward`               | ✅   | —                                |
| T | Comparator matrix                      | eq. 1 : `C = X - Ŷ`                              | residual computation in `testing()` L806                                | `ComparatorMatrix.forward` → `X - reconstruction`         | ✅   | —                                |
| U | Wager head                             | eq. 3 : 2 raw logits                              | 2-unit linear                                                           | `WageringHead` — **1-unit sigmoid** (D-001 already open)  | ❌   | D-001 (already open)            |
| V | Main-task loss                         | eq. 4 : contrastive                               | CAE (same as SARL — D-002)                                              | Same CAE                                                  | 🆘+❌ | D-002 (already open)            |
| W | **Evaluation metric**                  | Table 5 "Main Task Acc" = 0.97 (paper §Results p. 12) | `testing()` L806 — returns `discrimination_performances`, `f1_scores_wager` | `BlindsightTrainer.evaluate()` L344 — returns `discrimination_accuracy`, `wager_accuracy` | ⚠️🚨 | D-blindsight-metric-mismatch    |

**7 architectural rows. Divergences : 2 héritées (D-001 wager 1-unit sigmoid, D-002 CAE vs contrastive) + 1 🚨 nouvelle (metric mismatch — RG-002 cause probable).**

### 3. Seeds

| Source                               | Seeds                                                    |
|--------------------------------------|:--------------------------------------------------------:|
| Paper Table 9 preamble               | **500** *("training over the 500 seeds took roughly 12 hours")* |
| Student `main()` `seeds = 5`          | 5 (basic) / 10 (violin_seeds)                            |
| Our `experiment_matrix.md` (pre-B.13) | 10 ❌                                                     |

**🚨 Findings** : le student code ne fait qu'au max **10 seeds** (seeds_violin) — **pas 500**.
Le "500 seeds" du préambule Table 9 reste un mystère :

- Hypothèse (a) : aggregation cross-configs de la grid search (5 hidden × 3 step × 2 gamma × 5 seeds
  ≈ 150 ; or 10 seeds × 50 configs = 500). **Plausible.**
- Hypothèse (b) : une version du code non-vendored a une boucle seeds plus large.
- Hypothèse (c) : le préambule est une estimation marketing, pas une valeur exacte.

**À documenter comme `D-blindsight-seeds`** (Phase B.13) et décider Phase D.23 comment reproduire :
probablement 500 seeds littéral est raisonnable (Blindsight est petit, 12h paper).

### 4. Test conditions

Paper §A.1 p. 25 liste **3 conditions** de test :
1. **Suprathreshold** — familiar patterns, stimulus-present.
2. **Subthreshold** — noise increased (bruit ≈ stimulus).
3. **Low vision** — stimulus intensity reduced.

Table 5a (p. 13) **ne rapporte que suprathreshold**.

| Condition      | Paper data | Student `testing()` | Our port evaluate() | ∆ |
|----------------|:----------:|:-------------------:|:-------------------:|:-:|
| Suprathreshold | Table 5a row | computed            | computed            | ✅ |
| Subthreshold   | (paper silent on results)  | computed            | computed ?          | ⚠️ |
| Low vision     | (paper silent on results)  | computed            | computed ?          | ⚠️ |

À vérifier Phase D.24 : notre port génère-t-il les 3 conditions comme le student ?

### 5. RG-002 diagnostic (spécifique B.9)

**Gap mesuré** : reproduction Sprint-06 à N=10 seeds a donné :

| Metric                 | Paper (N=500) | Our measured (N=10) | Gap   |
|------------------------|:-------------:|:-------------------:|:-----:|
| Main Task Accuracy     | 0.97 ± 0.02   | **0.755** (discrimination)   | −22 % |
| Main Task Z            | 9.01          | **+0.40**            | 22× en-dessous |
| Wagering Accuracy      | 0.85 ± 0.04   | 0.71                | −16 % |

**3 hypothèses causales, ordonnées par plausibilité** :

1. **H1 — hidden_dim = 100 au lieu de 60** (D-blindsight-hidden-dim, row B3). Avec hidden=input=100,
   l'autoencoder n'a plus de goulot → reconstruction triviale → comparison matrix ≈ 0 → wager
   aveugle. Le student paper a sélectionné 60 après grid search parce que c'était le meilleur point
   signal/compression. **Fix proposé Phase D.25** : `config.first_order.hidden_dim = 60`.

2. **H2 — metric mismatch (D-blindsight-metric-mismatch, row W)**. Paper "Main Task Acc" vs notre
   "discrimination_accuracy". Notre `evaluate()` L413 :
   *"Discrimination accuracy on the stimulus-present portion."* — **c'est du recall, pas de
   l'accuracy overall.** Paper peut vouloir (TP+TN)/(TP+TN+FP+FN), pas TP/(TP+FN). Si le
   baseline (setting 1) a 50 % recall et 95 % accuracy (classifier qui dit toujours "present" →
   100 % recall 50 % accuracy ; ou baseline qui a appris → 95 % acc mais 75 % recall), le gap
   est une illusion métrique. **Fix proposé Phase D.25** : étendre `evaluate()` à reporter
   `overall_accuracy` en plus, et tester les deux.

3. **H3 — N trop petit** (N=10 vs paper 500). Avec N=10, la variance est haute → z=+0.40 peut
   être du bruit statistique. Repasser à N=500 requiert ~6-12h CPU local (paper : 12h pour 500
   sur RTX3070) ou qq heures GPU Tamia. **Fix Phase F.1**.

**Procédure de fix RG-002 proposée (Phase D.25)** : appliquer H1 + H2 d'abord (courtes), puis
re-runner à N=100 puis N=500 seulement si le gap subsiste. Si H1+H2 ne ferme pas le gap,
investiguer autres architectural choices (dropout, eval threshold, training loop).

### 6. Summary of Blindsight divergences

**Divergences identifiées Blindsight (8 distinctes) + héritées** :

| # | Deviation ID                        | Paper T.9 / text | Student                       | Port              | Verdict | Action                               |
|---|-------------------------------------|-----------------:|------------------------------:|------------------:|:-------:|:-------------------------------------|
| 1 | **D-blindsight-hidden-dim**         | **60**           | grid [30,40,50,60,100]        | **100** ❌         | 🆘+❌   | D.25 — fix to 60 (likely RG-002 cause H1) |
| 2 | **D-blindsight-metric-mismatch**    | "Main Task Acc" 0.97 | `testing()` reports similar  | `evaluate()` reports `discrimination_accuracy` = recall-only | 🚨 | D.25 — align metric definition (H2) |
| 3 | D-blindsight-seeds                  | 500              | 5-10                          | 10 (matrix)       | ⚠️      | D.23 + B.13 + F.1 — run N=500        |
| 4 | D-blindsight-temperature            | 1.0              | inferred                      | config silent     | ⚠️      | D.23 — confirm softmax temperature   |
| 5 | D-blindsight-epochs                 | 200              | grid runs                     | 200               | ⚠️      | D.23 — verify 200 applies            |
| 6 | D-blindsight-dropout-rate           | silent           | p=0.1 (wager) / p=0.5 (other) | per maps.yaml     | ⚠️      | D.23 — confirm match                 |
| 7 | D-001 (wager 1-unit vs 2-unit)     | 2 raw logits     | 2-unit                        | **1-unit sigmoid** | ❌     | already open, D.23 — re-audit |
| 8 | D-002 (contrastive-vs-CAE)          | SimCLR eq. 4     | CAE                           | CAE               | ❌+🆘   | already open, C.7-C.9                |

**Finding le plus important B.9** : **D-blindsight-hidden-dim** (port 100 vs paper 60) est une
divergence **silencieuse majeure** qui n'avait jamais été flagguée. **C'est la cause #1 suspectée
de RG-002.** La fix la plus probable : `hidden_dim: 60` + re-run N=10 local pour voir si
z-score passe de +0.40 à qqch de significatif avant d'aller jusqu'à N=500.

**Next** : Phase B.10 audit AGL (section suivante).

---
