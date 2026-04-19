# Paper tables — verbatim extractions

**Source :** `pdf/MAPS_TMLR_Journal_Submission.pdf` — Vargas et al., TMLR submission (2025), 30 pages.

Règle : **tout ce qui est reporté ici est copié verbatim du papier.** Pas de paraphrase, pas
d'arrondi, pas de réinterprétation. Quand le papier est ambigu ou incohérent avec lui-même (par
exemple Table 11 appendix vs valeurs du texte principal), les deux sources sont mentionnées avec
l'écart flaggé. Ce document est la source de vérité côté "papier" pour
`docs/reproduction/paper_vs_code_audit.md` et `docs/reproduction/deviations.md`.

Phase B du plan `docs/plans/plan-20260419-review-and-reproduce.md` remplit ce doc table par table
(B.1 → B.4).

---

## Table 11 — MinAtar hyperparameters (SARL + SARL+CL)

**Paper location :** Appendix B.3, p. 30.

### Preamble (verbatim)

> For the MinAtar environments, we used a GPU V100 for training. The training time was maximum for
> MAPS (2nd order network and cascade model in both 1st and 2nd order network). For this setting,
> training took roughly 6 days per million steps per seed, and double when training with our
> curriculum learning approach.

### Table 11 (verbatim, 24 data rows)

| Hyperparameter                                             | Value   |
|------------------------------------------------------------|--------:|
| Batch size                                                 | 128     |
| Replay buffer size                                         | 100,000 |
| Target network update frequency                            | 1,000   |
| Training frequency                                         | 1       |
| Number of frames                                           | 500,000 |
| First N frames                                             | 100,000 |
| Replay start size                                          | 5,000   |
| End epsilon                                                | 0.1     |
| Step size                                                  | 0.0003  |
| Step size (second order)                                   | 0.0002  |
| Gradient momentum                                          | 0.95    |
| Squared gradient momentum                                  | 0.95    |
| Minimum squared gradient                                   | 0.01    |
| Gamma                                                      | 0.999   |
| Step Size                                                  | 1       |
| Epsilon                                                    | 1.0     |
| Alpha                                                      | 0.45    |
| Cascade iterations                                         | 50      |
| Optimizer                                                  | Adam    |
| Max input channels (Continual learning)                    | 10      |
| weight task loss (Continual learning)                      | 0.3     |
| weight weight regularization loss (Continual learning)     | 0.6     |
| weight feature loss (Continual learning)                   | 0.1     |

*(1 header row + 24 data rows ; le "2nd Step Size = 1" (row 15) est bien un champ distinct du "Step
size = 0.0003" (row 9), voir Known ambiguities.)*

### Known ambiguities — internes au papier

Incohérences détectées à la lecture, à documenter maintenant pour ne pas les perdre :

- **"Number of frames"** — Table 11 dit **500,000** ; main text p. 13 (SARL Results) dit :
  > *"Training occurs over 1 million steps across 3 seeds per configuration."*

  De plus le préambule ci-dessus dit *"roughly 6 days per million steps per seed"* ce qui suppose
  implicitement 1 M steps. **Écart 2× entre table appendix et texte.** Convention TMLR : l'appendix
  wins par défaut. À flagger dans `deviations.md` sous `D-sarl-num-frames` (Phase B.12).

- **Deux rows nommées "Step size"** — row 9 (`0.0003`) et row 15 (`1`). Sémantiquement distinctes :
  - Row 9 = `lr_first_order` (learning rate Adam de la main task network).
  - Row 15 = `StepLR.step_size` (période de décroissance du scheduler, en epochs/updates).
  Confirmé contre `external/paper_reference/sarl_maps.py` (ligne 98 : `step_size1 = 0.0003` ;
  ligne 1314 : `StepLR(optimizer, step_size=1000, ...)`). **Le papier appelle deux choses
  différentes "Step size" sans les distinguer** → source de confusion au merge ou au diff de config.

- **"Alpha" = 0.45** — désigne le facteur de lissage EMA du wagering target (eq. 13 du papier,
  p. 8). **NE PAS confondre** avec l'α du cascade model (= 1/`Cascade iterations` = 1/50 = 0.02,
  main text p. 6 : *"We empirically selected 50 cascade iterations for all test cases (α = 0.02)"*).
  Deux α différents dans le papier avec le même nom.

- **"Epsilon" = 1.0** — valeur de départ de l'ε-greedy exploration, **annealed** vers la valeur
  "End epsilon = 0.1" sur les premières "First N frames = 100,000". La table donne les trois
  valeurs (start, end, horizon) sans les relier explicitement.

- **"Optimizer = Adam"** — confirmé par Table 11 et par `sarl_maps.py` ligne 1313
  (`optim.Adam(policy_net.parameters(), ...)`). Toutefois des docstrings + commentaires internes du
  code student (ligne 508 : *"optimizer: centered RMSProp"*) mentionnent RMSprop, vestiges d'un
  ancien pipeline. Aucun appel RMSprop n'est réellement fait. À noter mais non-bloquant.

### Interpretation notes

Décodage minimal pour les phases d'audit suivantes (Phase B.7) :

- `Gradient momentum = 0.95` = **Adam β₁** (défaut PyTorch = 0.9). Paper diverge du défaut.
- `Squared gradient momentum = 0.95` = **Adam β₂** (défaut PyTorch = 0.999). Paper diverge du défaut.
- `Minimum squared gradient = 0.01` = **Adam `eps`** (défaut PyTorch = `1e-8`). Paper diverge
  largement du défaut. Déjà aligné dans notre `config/training/sarl.yaml` → `optimizer.eps = 0.01`.
- `Max input channels (CL) = 10` = l'union des channels de tous les jeux MinAtar couverts par la
  curriculum learning (SpaceInvaders=6, Breakout=4, Seaquest=10, Asterix=4, Freeway=7 → max=10).
  Canaux manquants zero-padded + conv 1×1 d'adaptation, main text p. 9.
- `weight task / reg / feature = 0.3 / 0.6 / 0.1` — les λ de L_continual (eq. 18, p. 9). Somme = 1,
  normalisés. **Le texte p. 17 mentionne un "optimal" différent** `0.4 / 0.4 / 0.2` — autre
  ambiguïté interne papier, à flagger.

---

## Table 9 — Blindsight hyperparameters

**Paper location :** Appendix B.1, p. 28.

### Preamble (verbatim)

> For the blindsight task, we used a Nvidia RTX3070 gpu for training, with 8GB of RAM. The training
> time was maximum for MAPS (2nd order network and cascade model in both 1st and 2nd order network).
> For this setting, training over the 500 seeds took roughly 12 hours.

### Table 9 (verbatim, 11 data rows)

| Hyperparameter                  | Value   |
|---------------------------------|--------:|
| Input size                      | 100     |
| Output size                     | 100     |
| Hidden size                     | 60      |
| lr first order                  | 0.5     |
| lr second order                 | 0.1     |
| Temperature                     | 1.0     |
| Step size                       | 25      |
| Gamma                           | 0.98    |
| Epochs number for training      | 200     |
| Optimizer                       | Adamax  |
| Cascade iterations              | 50      |

### Known ambiguities

Aucune incohérence interne détectée à la lecture de Table 9. La table est autosuffisante et ses
valeurs sont reprises à l'identique dans le main text §4 "Blindsight Results" (page 12-13).

### Interpretation notes

Décodage minimal pour Phase B.9 (audit Blindsight) :

- `Input size = Output size = 100` → Blindsight est un **autoencoder** sur un vecteur 100-dim
  (paper §A.1 p. 25 : *"400 patterns, equally split between two types: Random noise patterns
  (0.0–0.02 activations) and Designed stimulus patterns (each pattern includes one unit with
  0.0–1.0 activation)"*). Un pattern = 100 dimensions.
- `Hidden size = 60` — valeur **Blindsight-specific**, distincte de la constante MAPS AGL
  (`first_order_hidden_dim = 40` selon CLAUDE.md). Pas une déviation : chaque domaine a son hidden
  dim.
- `lr first order = 0.5` — lr très élevé. L'optimizer est **Adamax** (Kingma & Ba 2015 §8), qui
  tolère des lr plus grands que Adam grâce à la normalisation infinity-norm. Pas une erreur, c'est
  une caractéristique d'Adamax.
- `Step size = 25` + `Gamma = 0.98` = `StepLR` scheduler (période en epochs × décroissance multiplicative).
  Cohérent avec les constantes canoniques MAPS dans `config/maps.yaml` (`scheduler_step = 25`,
  `scheduler_gamma = 0.98`).
- `Epochs number for training = 200` — Blindsight s'entraîne en un seul passage (vs AGL qui a une
  pre-train + training split — voir Table 10).
- `Cascade iterations = 50` ⇒ α_cascade = 1/50 = 0.02, cohérent paper main text p. 6.
- `Temperature = 1.0` — paramètre du softmax final de l'autoencoder, à confirmer côté `blindsight_tmlr.py`.
- `Optimizer = Adamax` — **pas Adam**, contrairement à SARL/MARL. Blindsight + AGL utilisent des
  optimizers différents des RL tasks (plus grands lr + optimizers spéciaux).
- **N = 500 seeds** par le préambule (*"training over the 500 seeds took roughly 12 hours"*) —
  valeur à refléter dans `experiment_matrix.md` (Phase B.13), qui pour l'instant dit N=10 et est
  donc WRONG.

---

## Table 10 — AGL hyperparameters

**Paper location :** Appendix B.2, p. 29.

### Preamble (verbatim)

> For the AGL task, we used a Nvidia RTX3070 gpu for training, with 8GB of RAM. The training time
> was maximum for MAPS (2nd order network and cascade model in both 1st and 2nd order network).
> For this setting, training over the 500 seeds took roughly 12 hours.

### Table 10 (verbatim, 13 data rows)

| Hyperparameter                                     | Value     |
|----------------------------------------------------|----------:|
| Input size                                         | 48        |
| Output size                                        | 48        |
| Hidden size                                        | 40        |
| lr first order                                     | 0.4       |
| lr second order                                    | 0.1       |
| Temperature                                        | 1.0       |
| Step size                                          | 1         |
| Gamma                                              | 0.999     |
| Epochs number for pre-training                     | 60        |
| Epochs number for training (high consciousness)    | 12        |
| Epochs number for training (low consciousness)     | 3         |
| Optimizer                                          | RangerVA  |
| Cascade iterations                                 | 50        |

### Known ambiguities

- **Optimizer "RangerVA"** n'est pas dans `torch.optim` standard. C'est un optimizer communautaire
  (hybride RAdam + LookAhead + Gradient Centralization, implémentation Wright 2020 sous la PyPI
  package `ranger-adabelief` ou `pytorch-ranger`). Le student `external/paper_reference/agl_tmlr.py`
  l'importe via `torch_optimizer as optim2`. À confirmer Phase B.10 quel est le mapping exact.
- **Step size = 1 + Gamma = 0.999** : contrairement à Blindsight (`step_size=25, gamma=0.98`), AGL
  décroît son lr à **chaque epoch** (step_size=1) de 0.1% (gamma=0.999). Cela veut dire qu'après
  60 epochs de pre-training le lr est multiplié par 0.999^60 ≈ 0.94 — décroissance douce.

### Interpretation notes

Décodage minimal pour Phase B.10 (audit AGL) + Phase D.28 (port RG-003) :

- `Input size = Output size = 48` → AGL autoencoder sur vecteur 48-dim. Paper §A.2 p. 25 : chaînes
  de 3 à 8 lettres, encodage winner-takes-all par chunks de 6 bits ; 8 letters × 6 bits = 48.
- `Hidden size = 40` → correspond à la constante MAPS canonique `first_order_hidden_dim = 40` dans
  `config/maps.yaml` (CLAUDE.md). **OK, aligné.**
- `lr first order = 0.4` — large. L'optimizer **RangerVA** (RAdam + LookAhead + GradCentralization)
  tolère des lr élevés grâce à ses corrections adaptatives.
- **3-phase training** (c'est la clé de RG-003) :
  1. **Pre-training 60 epochs** — entraîne le second-order sur random grammars (paper §Results AGL
     p. 13 : *"we pre-train the model, save the weights of the second-order network, and disable
     backpropagation through it during training for testing."*). Notre port fait cette phase.
  2. **Training (high awareness) 12 epochs** — supervisé sur grammar A vs B **avec** second-order
     frozen. Paper dit que c'est le régime "explicit" (learned rule awareness).
  3. **Training (low awareness) 3 epochs** — idem mais plus court. Régime "implicit".
  Notre port (`src/maps/experiments/agl/trainer.py`) **ne fait que la phase 1** → classification
  accuracy reste ~chance (0.07–0.09 mesurée vs 0.66 / 0.62 papier). **C'est RG-003**, Phase D.28.
- `Cascade iterations = 50` ⇒ α_cascade = 0.02, constante MAPS.
- `Temperature = 1.0` — softmax temperature sur la sortie autoencoder (chunk-wise, WTA par 6 bits).
- **N = 500 seeds** par le préambule — à propager dans `experiment_matrix.md` Phase B.13.

---

## Table 12 — MeltingPot (MARL) hyperparameters

**Paper location :** Appendix B.4, p. 30.

### Preamble (verbatim)

> For the meltingpot tasks, we used a Nvidia A100 gpu for training. The average training time was
> roughly 16 hours per seed (baseline, MAPS not implemented fully, only with simple 2nd order
> network with no cascade model due to limitations with computational resources). Every run
> required roughly 4-6 GB of RAM, mainly depending on the number of agents.

### Table 12 (verbatim, 13 data rows)

| Hyperparameter                    | Value  |
|-----------------------------------|-------:|
| Num agents (harvest closed)       | 6      |
| Num agents (harvest partnership)  | 4      |
| Num agents (chemistry)            | 8      |
| Num agents (territory)            | 5      |
| Hidden size                       | 100    |
| Actor lr                          | 7e-5   |
| Critic lr                         | 100    |
| Num env steps                     | 15e6   |
| Entropy coef                      | 0.01   |
| Clip param                        | 0.2    |
| Weight decay                      | 1e-5   |
| PPO epoch                         | 15     |
| Optimizer                         | Adam   |

*(1 header + 13 data rows. Plan B.4 Vérifier disait 12 ; recomptage paper Table 12 = 13 comme pour
B.1. Off-by-one plan à corriger dans le changelog.)*

### Known ambiguities — internes au papier

Deux flags critiques à lever avant tout port ou config alignment (Phase B.11, E.17) :

- ⚠️ **`Critic lr = 100`** — **quasi-certainement un typo**. Aucun sens sous Adam (`lr=100` diverge
  immédiatement). Hypothèses plausibles :
  - `7e-3` (= 100 × 7e-5, ratio actor-critic usuel en MAPPO — Yu et al. 2022 default `critic_lr=7e-3`)
  - `1e-2` (ratio 142× vs actor)
  - Perte d'un suffixe "e-3" au copy-paste vers LaTeX
  À **vérifier** contre `MARL/MAPPO-ATTENTIOAN/` Phase B.11 pour arbitrer.
- ⚠️ **`Num env steps = 15e6` (15 millions)** — **conflit main text**. Paper page 15, MARL Results :
  > *"The agents were trained for 300,000 steps on three seeds, due to computational constraints
  > (refer to Table 4)."*

  Écart **50×**. Incohérence interne du papier :
  1. **15e6 est faux** : A100 × 15M steps × 84 cells ≈ semaines, incompatible avec le "16h per seed"
     du preamble lui-même.
  2. **300k est faux** : possible mais très peu plausible (un ordre de magnitude entre table et
     texte sur le même domaine c'est rare).
  Hypothèse forte : **table wrong, texte right → 300,000 est la bonne valeur**. À confirmer via
  `MARL/MAPPO-ATTENTIOAN/` shells ou code Phase B.11.
- **MAPS "not implemented fully"** (preamble) — paper avoue que le 2nd-order MARL n'a pas de cascade
  model faute de compute. Implication : reproduction MARL **ne pourra pas** être strictement
  paper-faithful pour les settings qui utilisent cascade (2, 4, 5, 6). À documenter dans
  `deviations.md` Phase B.12 comme **limitation déclarée par le papier lui-même**.
- **"Optimizer = Adam"** — confirmé. Pas RangerVA ni Adamax ici.

### Interpretation notes

Pour Phase B.11 (audit MARL) + Phase E.11-E.17 (port MARL) :

- **4 Num agents distincts** par substrate (harvest_closed=6, harvest_partnership=4, chemistry=8,
  territory=5). Le code devra dispatcher la valeur au bon substrate au runtime.
- `Hidden size = 100` = constante MAPS canonique `second_order_hidden_dim = 100` (CLAUDE.md).
  Cohérent cross-domain.
- `Actor lr = 7e-5` — faible, standard pour MAPPO large-scale (Yu et al. 2022).
- `Critic lr = 100` — **à ignorer** jusqu'à arbitrage Phase B.11 ; ne **pas** aligner notre config
  sur cette valeur littérale.
- `Num env steps` — **idem**, utiliser 300k du main text jusqu'à arbitrage.
- `Entropy coef = 0.01` — terme d'entropy bonus classique PPO, encourage l'exploration.
- `Clip param = 0.2` — PPO standard (Schulman et al. 2017).
- `Weight decay = 1e-5` — L2 reg actor+critic, faible.
- `PPO epoch = 15` — nombre de passes gradient par rollout (plus élevé que le 10 default PPO).
- **N = 3 seeds** (main text p. 15, confirmé pour SARL/SARL+CL/MARL) — **pas 10** (nos docs actuels).
  À propager Phase B.13.

---
