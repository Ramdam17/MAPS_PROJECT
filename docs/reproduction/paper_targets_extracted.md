# Paper reproduction targets — Tables 5, 6, 7 verbatim

**Source :** `pdf/MAPS_TMLR_Journal_Submission.pdf` — Vargas et al., TMLR submission (2025).

Règle : chiffres copiés **verbatim**, y compris les z-scores négatifs (quand MAPS ne bat pas le
baseline), les "(No)" de non-significativité, et les `NA` de la table 7. **Aucune mise en forme
créative** : ces nombres sont les cibles de reproduction à ±2σ de la Phase F du plan
`docs/plans/plan-20260419-review-and-reproduce.md`.

---

## Convention seeds (source : preambles Appendix B + main text)

| Domain                    | N seeds |
|---------------------------|:-------:|
| Blindsight                | **500** |
| AGL                       | **500** |
| SARL / SARL+CL / MARL     | **3**   |

Les docs actuels `docs/reproduction/experiment_matrix.md` et le plan Sprint-07 verrouillaient N=10
pour les tâches RL — **WRONG**. Paper says N=3. Correction à propager Phase B.13.

Confidence : 95 % (paper text, Tables 5/6/7 captions). Significance notation `(Yes)` / `(No)`.

---

## Table 5 — Know-Thyself (Blindsight + AGL)

**Paper location :** p. 13.

**Caption (verbatim) :**

> Accuracy, Z-score, and Significant Results for Main Task and Wagering (Know Thyself environments).
> Z-score is a statistical measure that quantifies the distance between a data point and the mean
> of a dataset. We use 95 percent confidence to establish statistical significance. We use a total
> of N = 500 seeds for each setting. 2nd Net refers to the presence (or not) of a second-order
> network.

Le paper présente uniquement la condition **suprathreshold** pour Blindsight (celle sur laquelle
"metacognition should be beneficial", Weiskrantz 1974).

### Table 5a — Blindsight (suprathreshold)

| Setting               | 2nd Net | Cascade  | Main Task Acc | Main Z (Signif) | Wagering Acc | Wagering Z (Signif) |
|-----------------------|:-------:|:--------:|--------------:|----------------:|-------------:|--------------------:|
| Setting-1 (Baseline)  | No      | No       | 0.95 ± 0.03   | —               | 0.50 ± 0.05  | —                   |
| Setting-2             | No      | 1st Net  | 0.97 ± 0.02   | **8.50 (Yes)**  | 0.50 ± 0.05  | 0.45 (No)           |
| Setting-3             | Yes     | No       | 0.96 ± 0.03   | 0.77 (No)       | 0.86 ± 0.03  | **128.1 (Yes)**     |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 0.97 ± 0.02   | **9.01 (Yes)**  | 0.85 ± 0.04  | **121.2 (Yes)**     |
| Setting-5             | Yes     | 2nd Net  | 0.96 ± 0.03   | 0.15 (No)       | 0.87 ± 0.04  | **126.7 (Yes)**     |
| Setting-6             | Yes     | Both     | 0.97 ± 0.02   | **8.6 (Yes)**   | 0.86 ± 0.04  | **124.5 (Yes)**     |

### Table 5b — AGL High Awareness

| Setting               | 2nd Net | Cascade  | Main Task Acc | Main Z (Signif) | Wagering Acc | Wagering Z (Signif) |
|-----------------------|:-------:|:--------:|--------------:|----------------:|-------------:|--------------------:|
| Setting-1 (Baseline)  | No      | No       | 0.63 ± 0.05   | —               | 0.38 ± 0.07  | —                   |
| Setting-2             | No      | 1st Net  | 0.64 ± 0.04   | **6.38 (Yes)**  | 0.39 ± 0.09  | 1.10 (No)           |
| Setting-3             | Yes     | No       | 0.64 ± 0.04   | 1.61 (No)       | 0.59 ± 0.06  | **45.9 (Yes)**      |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 0.66 ± 0.05   | **8.20 (Yes)**  | 0.58 ± 0.06  | **43.3 (Yes)**      |
| Setting-5             | Yes     | 2nd Net  | 0.63 ± 0.04   | 1.09 (No)       | 0.61 ± 0.06  | **48.7 (Yes)**      |
| Setting-6             | Yes     | Both     | 0.65 ± 0.04   | **7.88 (Yes)**  | 0.58 ± 0.06  | **41.0 (Yes)**      |

### Table 5c — AGL Low Awareness

| Setting               | 2nd Net | Cascade  | Main Task Acc | Main Z (Signif) | Wagering Acc | Wagering Z (Signif) |
|-----------------------|:-------:|:--------:|--------------:|----------------:|-------------:|--------------------:|
| Setting-1 (Baseline)  | No      | No       | 0.54 ± 0.08   | —               | 0.14 ± 0.07  | —                   |
| Setting-2             | No      | 1st Net  | 0.61 ± 0.07   | **13.3 (Yes)**  | 0.17 ± 0.07  | **6.25 (Yes)**      |
| Setting-3             | Yes     | No       | 0.57 ± 0.07   | **4.2 (Yes)**   | 0.83 ± 0.07  | **143.9 (Yes)**     |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 0.62 ± 0.07   | **15.7 (Yes)**  | 0.82 ± 0.07  | **137.5 (Yes)**     |
| Setting-5             | Yes     | 2nd Net  | 0.56 ± 0.07   | **2.3 (Yes)**   | 0.87 ± 0.07  | **150.8 (Yes)**     |
| Setting-6             | Yes     | Both     | 0.62 ± 0.06   | **15.0 (Yes)**  | 0.82 ± 0.07  | **141.1 (Yes)**     |

**Total Table 5 : 18 data rows (3 sous-tables × 6 settings).**

---

## Table 6 — SARL (MinAtar)

**Paper location :** p. 15.

**Caption (verbatim) :**

> Training and validation rewards, Z-score, and significant results for SARL (MinAtar). Z-score is
> a statistical measure that quantifies the distance between a data point and the mean of a dataset.
> We use 95 percent confidence to establish statistical significance.

**N = 3 seeds** (paper text p. 15 : *"Training occurs over 1 million steps across 3 seeds per
configuration"*).

**Setting 7 = ACB** (Actor-Critic Baseline, Young & Tian 2019, trace decay λ = 0.8). Absent de notre
factorielle courante — à porter Phase E.1-E.5.

### Table 6a — Seaquest

| Setting               | 2nd Net | Cascade  | Training R.   | Train Z (Signif) | Validation R. | Val Z (Signif)   |
|-----------------------|:-------:|:--------:|--------------:|-----------------:|--------------:|-----------------:|
| Setting-1 (Baseline)  | No      | No       | 1.48 ± 0.29   | —                | 1.21 ± 0.16   | —                |
| Setting-2             | No      | 1st Net  | 0.90 ± 0.21   | **-2.32 (Yes)**  | 0.76 ± 0.19   | **-2.59 (Yes)**  |
| Setting-3             | Yes     | No       | 1.04 ± 0.57   | -0.97 (No)       | 0.97 ± 0.61   | -0.53 (No)       |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 3.71 ± 0.40   | **6.46 (Yes)**   | 3.06 ± 0.34   | **7.03 (Yes)**   |
| Setting-5             | Yes     | 2nd Net  | 4.10 ± 0.86   | **1.97 (Yes)**   | 3.99 ± 1.84   | 2.14 (Yes)       |
| Setting-6             | Yes     | Both     | 5.56 ± 1.85   | **3.09 (Yes)**   | 6.15 ± 2.34   | **2.98 (Yes)**   |
| Setting-7 (ACB)       | No      | No       | 0.81 ± 0.03   | **-3.26 (Yes)**  | 0.63 ± 0.26   | **-2.65 (Yes)**  |

### Table 6b — Asterix

| Setting               | 2nd Net | Cascade  | Training R.   | Train Z (Signif) | Validation R. | Val Z (Signif)   |
|-----------------------|:-------:|:--------:|--------------:|-----------------:|--------------:|-----------------:|
| Setting-1 (Baseline)  | No      | No       | 3.49 ± 3.32   | —                | 2.49 ± 1.94   | —                |
| Setting-2             | No      | 1st Net  | 1.38 ± 0.67   | -0.88 (No)       | 1.59 ± 0.90   | -0.60 (No)       |
| Setting-3             | Yes     | No       | 6.27 ± 1.87   | 1.03 (No)        | 5.48 ± 1.30   | 1.81 (No)        |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 4.95 ± 1.05   | 0.59 (No)        | 4.54 ± 1.01   | 1.32 (No)        |
| Setting-5             | Yes     | 2nd Net  | 3.40 ± 0.30   | -0.04 (No)       | 2.94 ± 0.25   | 0.32 (No)        |
| Setting-6             | Yes     | Both     | 5.42 ± 0.69   | 0.81 (No)        | 5.77 ± 0.94   | **2.15 (Yes)**   |
| Setting-7 (ACB)       | No      | No       | 2.75 ± 0.09   | -0.31 (No)       | 2.13 ± 1.07   | -0.23 (No)       |

### Table 6c — Breakout

| Setting               | 2nd Net | Cascade  | Training R.   | Train Z (Signif) | Validation R. | Val Z (Signif)   |
|-----------------------|:-------:|:--------:|--------------:|-----------------:|--------------:|-----------------:|
| Setting-1 (Baseline)  | No      | No       | 5.96 ± 0.43   | —                | 6.10 ± 0.21   | —                |
| Setting-2             | No      | 1st Net  | 7.08 ± 0.65   | **2.05 (Yes)**   | 6.84 ± 0.41   | **2.25 (Yes)**   |
| Setting-3             | Yes     | No       | 6.12 ± 0.51   | 0.36 (No)        | 6.22 ± 0.69   | 0.24 (No)        |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 8.54 ± 0.07   | **8.36 (Yes)**   | 8.07 ± 0.72   | **3.70 (Yes)**   |
| Setting-5             | Yes     | 2nd Net  | 6.23 ± 0.14   | 0.85 (No)        | 6.16 ± 0.05   | 0.39 (No)        |
| Setting-6             | Yes     | Both     | 8.08 ± 0.24   | **6.06 (Yes)**   | 7.91 ± 1.36   | 1.87 (No)        |
| Setting-7 (ACB)       | No      | No       | 12.36 ± 1.13  | **7.49 (Yes)**   | 11.67 ± 2.70  | **2.90 (Yes)**   |

### Table 6d — Space Invaders

| Setting               | 2nd Net | Cascade  | Training R.   | Train Z (Signif) | Validation R. | Val Z (Signif)   |
|-----------------------|:-------:|:--------:|--------------:|-----------------:|--------------:|-----------------:|
| Setting-1 (Baseline)  | No      | No       | 23.62 ± 0.86  | —                | 19.71 ± 1.84  | —                |
| Setting-2             | No      | 1st Net  | 31.52 ± 1.01  | **8.46 (Yes)**   | 29.62 ± 2.28  | **4.79 (Yes)**   |
| Setting-3             | Yes     | No       | 21.14 ± 2.41  | -1.37 (No)       | 20.95 ± 2.60  | 0.55 (No)        |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 26.84 ± 0.11  | **5.28 (Yes)**   | 26.80 ± 1.59  | **4.13 (Yes)**   |
| Setting-5             | Yes     | 2nd Net  | 28.11 ± 0.24  | **7.14 (Yes)**   | 27.95 ± 1.10  | **5.45 (Yes)**   |
| Setting-6             | Yes     | Both     | 26.57 ± 0.72  | **3.74 (Yes)**   | 22.97 ± 0.89  | **2.26 (Yes)**   |
| Setting-7 (ACB)       | No      | No       | 71.50 ± 17.38 | **3.89 (Yes)**   | 59.26 ± 16.61 | **3.34 (Yes)**   |

### Table 6e — Freeway

| Setting               | 2nd Net | Cascade  | Training R.    | Train Z (Signif) | Validation R. | Val Z (Signif)   |
|-----------------------|:-------:|:--------:|---------------:|-----------------:|--------------:|-----------------:|
| Setting-1 (Baseline)  | No      | No       | 30.96 ± 5.35   | —                | 29.03 ± 7.12  | —                |
| Setting-2             | No      | 1st Net  | 32.77 ± 8.23   | 0.26 (No)        | 32.57 ± 8.82  | 0.44 (No)        |
| Setting-3             | Yes     | No       | 24.72 ± 17.58  | -0.48 (No)       | 25.00 ± 17.68 | -0.30 (No)       |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 35.53 ± 0.08   | 1.21 (No)        | 34.20 ± 2.83  | 0.95 (No)        |
| Setting-5             | Yes     | 2nd Net  | 39.41 ± 1.89   | **2.11 (Yes)**   | 34.40 ± 2.97  | 0.98 (No)        |
| Setting-6             | Yes     | Both     | 33.84 ± 2.03   | 0.71 (No)        | 30.63 ± 2.36  | 0.30 (No)        |
| Setting-7 (ACB)       | No      | No       | 0.19 ± 0.10    | **-8.14 (Yes)**  | 0.17 ± 0.05   | **-5.73 (Yes)**  |

**Total Table 6 : 35 data rows (5 jeux × 7 settings).**

---

## Table 7 — MARL (MeltingPot 2.0)

**Paper location :** p. 17.

**Caption (verbatim) :**

> Training rewards, dist entropy results, Z-score, and significant results for MARL (Melting Pot 2.0).
> Dist entropy is the action distribution entropy, which gets lower as the agents reduce their
> stochastic behavior and start to behave in a more stable manner (dist entropy should reduce as
> rewards increase). Z-score is a statistical measure that quantifies the distance between a data
> point and the mean of a dataset. We use 95 percent confidence to establish statistical significance.

**N = 3 seeds** (paper text p. 15 : *"The agents were trained for 300,000 steps on three seeds,
due to computational constraints"*).

**Setting 7 (ACB) a `NA` sur Dist Entropy** — ACB ne produit pas cette métrique.

### Table 7a — Harvest Partnership

| Setting               | 2nd Net | Cascade  | Training R.    | Train Z (Signif) | Dist Entropy  | Dist Z (Signif)   |
|-----------------------|:-------:|:--------:|---------------:|-----------------:|--------------:|------------------:|
| Setting-1 (Baseline)  | No      | No       | 29.21 ± 0.72   | —                | 1.82 ± 0.07   | —                 |
| Setting-2             | No      | 1st Net  | 27.15 ± 0.04   | **-4.07 (Yes)**  | 1.71 ± 0.06   | -1.73 (No)        |
| Setting-3             | Yes     | No       | 29.03 ± 0.29   | -0.34 (No)       | 1.88 ± 0.03   | 1.04 (No)         |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 34.52 ± 0.98   | **6.20 (Yes)**   | 1.64 ± 0.10   | **-2.20 (Yes)**   |
| Setting-5             | Yes     | 2nd Net  | 28.63 ± 0.68   | -1.03 (No)       | 1.88 ± 0.05   | 0.93 (No)         |
| Setting-6             | Yes     | Both     | 33.01 ± 2.57   | **2.01 (Yes)**   | 1.62 ± 0.11   | **-2.28 (Yes)**   |
| Setting-7 (ACB)       | No      | No       | 24.58 ± 10.56  | -0.75 (No)       | NA            | NA                |

### Table 7b — Harvest Closed

| Setting               | 2nd Net | Cascade  | Training R.    | Train Z (Signif)  | Dist Entropy  | Dist Z (Signif)   |
|-----------------------|:-------:|:--------:|---------------:|------------------:|--------------:|------------------:|
| Setting-1 (Baseline)  | No      | No       | 19.52 ± 0.71   | —                 | 1.89 ± 0.01   | —                 |
| Setting-2             | No      | 1st Net  | 26.05 ± 0.71   | **9.18 (Yes)**    | 1.72 ± 0.01   | **-20.93 (Yes)**  |
| Setting-3             | Yes     | No       | 19.15 ± 0.42   | -0.64 (No)        | 1.86 ± 0.01   | **-4.51 (Yes)**   |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 25.21 ± 1.06   | **6.31 (Yes)**    | 1.76 ± 0.12   | -1.64 (No)        |
| Setting-5             | Yes     | 2nd Net  | 18.81 ± 0.68   | -1.02 (No)        | 1.86 ± 0.05   | -0.86 (No)        |
| Setting-6             | Yes     | Both     | 23.97 ± 1.30   | **4.25 (Yes)**    | 1.77 ± 0.01   | **-15.96 (Yes)**  |
| Setting-7 (ACB)       | No      | No       | 39.39 ± 10.69  | **3.19 (Yes)**    | NA            | NA                |

### Table 7c — Chemistry (Three Metabolic Cycles with Plentiful Distractors)

| Setting               | 2nd Net | Cascade  | Training R.    | Train Z (Signif) | Dist Entropy  | Dist Z (Signif)   |
|-----------------------|:-------:|:--------:|---------------:|-----------------:|--------------:|------------------:|
| Setting-1 (Baseline)  | No      | No       | 1.18 ± 0.09    | —                | 1.95 ± 0.01   | —                 |
| Setting-2             | No      | 1st Net  | 1.13 ± 0.01    | -0.70 (No)       | 1.81 ± 0.05   | **-3.61 (Yes)**   |
| Setting-3             | Yes     | No       | 1.25 ± 0.01    | 1.07 (No)        | 1.93 ± 0.02   | -1.15 (No)        |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 1.11 ± 0.01    | -0.91 (No)       | 1.83 ± 0.12   | -1.33 (No)        |
| Setting-5             | Yes     | 2nd Net  | 1.23 ± 0.06    | 0.72 (No)        | 1.90 ± 0.01   | **-3.66 (Yes)**   |
| Setting-6             | Yes     | Both     | 1.15 ± 0.03    | -0.37 (No)       | 1.81 ± 0.19   | -1.03 (No)        |
| Setting-7 (ACB)       | No      | No       | 1.41 ± 0.60    | 0.64 (No)        | NA            | NA                |

### Table 7d — Territory Inside Out

| Setting               | 2nd Net | Cascade  | Training R.    | Train Z (Signif) | Dist Entropy  | Dist Z (Signif)   |
|-----------------------|:-------:|:--------:|---------------:|-----------------:|--------------:|------------------:|
| Setting-1 (Baseline)  | No      | No       | 57.94 ± 6.84   | —                | 1.98 ± 0.00   | —                 |
| Setting-2             | No      | 1st Net  | 41.98 ± 7.43   | **-2.24 (Yes)**  | 2.01 ± 0.25   | 0.18 (No)         |
| Setting-3             | Yes     | No       | 54.04 ± 0.34   | -0.81 (No)       | 1.97 ± 0.02   | -0.66 (No)        |
| Setting-4 (MAPS)      | Yes     | 1st Net  | 48.47 ± 1.45   | -1.92 (No)       | 1.58 ± 0.08   | **-7.16 (Yes)**   |
| Setting-5             | Yes     | 2nd Net  | 58.83 ± 6.15   | 0.14 (No)        | 1.92 ± 0.07   | -1.29 (No)        |
| Setting-6             | Yes     | Both     | 46.05 ± 9.16   | -1.47 (No)       | 1.68 ± 0.12   | **-3.46 (Yes)**   |
| Setting-7 (ACB)       | No      | No       | 101.99 ± 38.67 | 1.89 (No)        | NA            | NA                |

**Total Table 7 : 28 data rows (4 substrates × 7 settings).**

---

## Headline targets for reproduction

Ce sont les **z-scores significatifs** (|z| ≥ 2, Signif = Yes) sur le Setting 4 (MAPS) qui doivent
être matchés à ±2σ. Plus quelques "supporting" z-scores sur settings 2/6 où le papier met en avant
MAPS ou son composant cascade.

### Domaines où MAPS gagne (z ≫ 2 ; à reproduire)

| Domain        | Setting / Condition               | Metric                           | Target mean ± std | Target Z    |
|---------------|-----------------------------------|----------------------------------|-------------------|-------------|
| Blindsight    | Setting-4 (MAPS) supratresh.      | Main Task Acc                    | 0.97 ± 0.02       | **9.01**    |
| Blindsight    | Setting-4 (MAPS) supratresh.      | Wagering Acc                     | 0.85 ± 0.04       | **121.2**   |
| AGL High      | Setting-4 (MAPS)                  | Main Task Acc                    | 0.66 ± 0.05       | **8.20**    |
| AGL High      | Setting-4 (MAPS)                  | Wagering Acc                     | 0.58 ± 0.06       | **43.3**    |
| AGL Low       | Setting-4 (MAPS)                  | Main Task Acc                    | 0.62 ± 0.07       | **15.7**    |
| AGL Low       | Setting-4 (MAPS)                  | Wagering Acc                     | 0.82 ± 0.07       | **137.5**   |
| SARL Seaquest | Setting-4 (MAPS) validation       | Mean Return                      | 3.06 ± 0.34       | **7.03**    |
| SARL Seaquest | Setting-4 (MAPS) training         | Mean Return                      | 3.71 ± 0.40       | **6.46**    |
| SARL Breakout | Setting-4 (MAPS) training         | Mean Return                      | 8.54 ± 0.07       | **8.36**    |
| SARL Breakout | Setting-4 (MAPS) validation       | Mean Return                      | 8.07 ± 0.72       | **3.70**    |
| SARL S. Inv.  | Setting-2 (cascade 1st Net)       | Training Return                  | 31.52 ± 1.01      | **8.46**    |
| SARL S. Inv.  | Setting-5 (cascade 2nd Net)       | Training Return                  | 28.11 ± 0.24      | **7.14**    |
| SARL S. Inv.  | Setting-4 (MAPS) validation       | Mean Return                      | 26.80 ± 1.59      | **4.13**    |
| MARL Harv. C. | Setting-2                         | Training Reward                  | 26.05 ± 0.71      | **9.18**    |
| MARL Harv. C. | Setting-4 (MAPS)                  | Training Reward                  | 25.21 ± 1.06      | **6.31**    |
| MARL Harv. P. | Setting-4 (MAPS)                  | Training Reward                  | 34.52 ± 0.98      | **6.20**    |

### Domaines "à la frontière du bruit" — à reporter honnêtement, pas à battre artificiellement

- **SARL Asterix, SARL Freeway** : MAPS z < 2 ou non-significatif. Paper reconnaît "at edge of noise".
- **MARL Chemistry, MARL Territory Inside Out** : MAPS z < 2 ou **négatif** (Chemistry setting 4
  z = -0.91, Territory setting 4 z = -1.92). Paper §5 p. 20 : *"MAPS' success appeared inversely
  correlated with environmental complexity—performing well in harvest environments (5-8 multi-agent
  concepts) but showing limited benefits in chemistry and territory scenarios (11-14 concepts)."*.
  Pas de tentative de "faire mieux" en Phase F.

### ACB (Setting 7) — comparison baseline, pas MAPS

Setting 7 existe pour comparer MAPS contre un baseline Actor-Critic fort (Young & Tian 2019).
**Parfois ACB bat MAPS** (ex: Breakout, Space Invaders sur le jeu raw) — paper est honnête là-dessus.
Reproduction Phase F doit inclure setting 7 pour fidélité, pas pour "gagner".

---

## Known gaps (reproduction à date, pre-Phase-F)

| ID         | Domain     | Target paper | Our measured (pre-Sprint-08) | Cause                                                                                      | Phase     |
|------------|------------|-------------:|-----------------------------:|--------------------------------------------------------------------------------------------|-----------|
| **RG-002** | Blindsight | z = **9.01** | z = +0.40 (discrim.) / +0.71 (wager) | Metric-definition mismatch probable (paper = aggregate across conditions ?)          | D.23-D.25 |
| **RG-003** | AGL        | z = **8.20** / **15.7** | z ≈ 0 (classif ~chance)      | **Downstream training phase non portée** dans `src/maps/experiments/agl/` (cf. Table 10)     | D.26-D.28 |
| (paper-declared) | MARL | — | — | Paper preamble Table 12 : *"MAPS not implemented fully, only with simple 2nd order network with no cascade model"*. Settings 2/4/5/6 MARL **ne peuvent pas** être strictement paper-faithful côté cascade. | B.12 (deviations) |

---
