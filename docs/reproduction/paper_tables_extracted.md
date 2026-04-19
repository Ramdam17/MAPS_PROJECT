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
