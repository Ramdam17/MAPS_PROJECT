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
