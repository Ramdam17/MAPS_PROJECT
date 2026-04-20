# Reproduction Target Matrix

**Source :** MAPS TMLR submission (Vargas et al., 2025) — `pdf/MAPS_TMLR_Journal_Submission.pdf`.
Z-scores papier extraits verbatim dans `docs/reproduction/paper_targets_extracted.md` (Phase B.6).

**Baseline de comparaison :** Setting 1 (second-order OFF + cascade OFF). Les z-scores paper Tables
5/6/7 sont calculés contre la moyenne/std de ce baseline, par domaine, à **95 % de confidence**.

**Tolérance de reproduction :** **±2σ** autour de la moyenne paper. Un domaine reproduit à ±2σ si
notre moyenne mesurée tombe dans `[paper_mean − 2·paper_std, paper_mean + 2·paper_std]`.

---

## Seed convention — paper vs plan Sprint-08 Phase F

| Domain           | Paper seeds | Plan Phase F |
|------------------|:-----------:|:------------:|
| Blindsight       | **500**     | 500          |
| AGL              | **500**     | 500          |
| SARL             | **3**       | 3            |
| SARL+CL          | **3**       | 3            |
| MARL             | **3**       | 3            |

Source : préambules Tables 9/10 pour perceptuel (*"training over the 500 seeds took roughly 12
hours"*), main text p.15 + captions Tables 6/7 pour RL (*"3 seeds per configuration"*).

**Correction B.13** : la version précédente de ce doc disait *"paper N=10 seeds"* — faux. Sprint-07
plan disait aussi N=10 → superseded par ce fichier + `deviations.md` entrées `D-*-seeds`.

---

## Settings factorial (1-7)

7 settings par domaine, confirmés contre paper Figure 6 (settings 1-6) + Tables 6/7 row Setting 7 :

| Setting | 2nd-order | Cascade           | Label                 | Source                                     |
|---------|:---------:|:-----------------:|:---------------------:|:-------------------------------------------|
| 1       | OFF       | OFF               | Baseline              | paper fig. 6                               |
| 2       | OFF       | 1st Net           | Cascade 1st only      | paper fig. 6                               |
| 3       | ON        | OFF               | 2nd-order no cascade  | paper fig. 6                               |
| 4       | ON        | 1st Net           | **MAPS**              | paper fig. 6                               |
| 5       | ON        | 2nd Net           | Cascade 2nd only      | paper fig. 6                               |
| 6       | ON        | Both              | **Full MAPS**         | paper fig. 6                               |
| 7       | —         | —                 | **ACB**               | paper Tables 6/7 (Young & Tian 2019 λ=0.8) |

- Setting 7 (ACB = Actor-Critic Baseline, trace decay λ=0.8) s'applique à **SARL et MARL
  uniquement** — Tables 5a-5c (Blindsight/AGL) n'ont que 6 rows.
- Pour MARL, paper preamble Table 12 indique *"MAPS not implemented fully, only with simple 2nd
  order network with no cascade model"* → settings 2/4/5/6 MARL ne sont **pas** paper-faithful
  côté cascade (limitation paper-declared, `D-marl-cascade-not-implemented`).

---

## 1. Blindsight — paper Table 5a (suprathreshold)

| Setting      | Main Task Acc | Main Z        | Wagering Acc | Wagering Z       |
|--------------|:-------------:|:-------------:|:------------:|:----------------:|
| 1 Baseline   | 0.95 ± 0.03   | —             | 0.50 ± 0.05  | —                |
| 2            | 0.97 ± 0.02   | **8.50 (Yes)**| 0.50 ± 0.05  | 0.45 (No)        |
| 3            | 0.96 ± 0.03   | 0.77 (No)     | 0.86 ± 0.03  | **128.1 (Yes)**  |
| **4 MAPS**   | 0.97 ± 0.02   | **9.01 (Yes)**| 0.85 ± 0.04  | **121.2 (Yes)**  |
| 5            | 0.96 ± 0.03   | 0.15 (No)     | 0.87 ± 0.04  | **126.7 (Yes)**  |
| 6 Full MAPS  | 0.97 ± 0.02   | **8.6 (Yes)** | 0.86 ± 0.04  | **124.5 (Yes)**  |

**Headline target** — Setting 4 : Main Task Acc 0.97 ± 0.02 (±2σ : [0.93, 1.01]), Main Z = 9.01.

**Current reproduction status (Sprint-06, N=10)** :
- Main Task Acc mesuré **0.755** (discrimination recall-only) vs paper **0.97** → z = **+0.40** vs
  paper 9.01.
- Wager Acc mesuré 0.71 vs paper 0.85 → gap secondaire.
- **Gap = RG-002**. Causes confirmées B.9 : `D-blindsight-hidden-dim` (port 100 vs paper 60) +
  `D-blindsight-metric-mismatch` (recall vs overall accuracy). Fix Phase D.25.

*Paper §A.1 définit aussi les conditions subthreshold + low vision, mais Table 5 ne reporte que
suprathreshold — c'est la target canonique.*

---

## 2. AGL — paper Table 5b (high awareness) + 5c (low awareness)

### 2a. High Awareness (12 training epochs, explicit condition)

| Setting      | Main Task Acc | Main Z        | Wagering Acc | Wagering Z       |
|--------------|:-------------:|:-------------:|:------------:|:----------------:|
| 1 Baseline   | 0.63 ± 0.05   | —             | 0.38 ± 0.07  | —                |
| 2            | 0.64 ± 0.04   | **6.38 (Yes)**| 0.39 ± 0.09  | 1.10 (No)        |
| 3            | 0.64 ± 0.04   | 1.61 (No)     | 0.59 ± 0.06  | **45.9 (Yes)**   |
| **4 MAPS**   | 0.66 ± 0.05   | **8.20 (Yes)**| 0.58 ± 0.06  | **43.3 (Yes)**   |
| 5            | 0.63 ± 0.04   | 1.09 (No)     | 0.61 ± 0.06  | **48.7 (Yes)**   |
| 6 Full MAPS  | 0.65 ± 0.04   | **7.88 (Yes)**| 0.58 ± 0.06  | **41.0 (Yes)**   |

### 2b. Low Awareness (3 training epochs, implicit condition)

| Setting      | Main Task Acc | Main Z        | Wagering Acc | Wagering Z       |
|--------------|:-------------:|:-------------:|:------------:|:----------------:|
| 1 Baseline   | 0.54 ± 0.08   | —             | 0.14 ± 0.07  | —                |
| 2            | 0.61 ± 0.07   | **13.3 (Yes)**| 0.17 ± 0.07  | **6.25 (Yes)**   |
| 3            | 0.57 ± 0.07   | **4.2 (Yes)** | 0.83 ± 0.07  | **143.9 (Yes)**  |
| **4 MAPS**   | 0.62 ± 0.07   | **15.7 (Yes)**| 0.82 ± 0.07  | **137.5 (Yes)**  |
| 5            | 0.56 ± 0.07   | **2.3 (Yes)** | 0.87 ± 0.07  | **150.8 (Yes)**  |
| 6 Full MAPS  | 0.62 ± 0.06   | **15.0 (Yes)**| 0.82 ± 0.07  | **141.1 (Yes)**  |

**Headline targets** :
- High Awareness Setting 4 : Main Task Acc 0.66 ± 0.05 (±2σ : [0.56, 0.76]), Main Z = **8.20**.
- Low Awareness  Setting 4 : Main Task Acc 0.62 ± 0.07 (±2σ : [0.48, 0.76]), Main Z = **15.7**.

**Current reproduction status (Sprint-06, N=10)** :
- High Awareness Main Task Acc mesuré **0.073** vs paper 0.66 → z ≈ 0.
- Low Awareness Main Task Acc mesuré **0.093** vs paper 0.62 → z ≈ 0.
- **Gap = RG-003**. Cause structurelle B.10 : `D-agl-training-missing` — la phase 2 supervised
  training sur grammar A (post-pretrain) **n'est pas portée**. Fix Phase D.28.

---

## 3. SARL — paper Table 6 (MinAtar)

5 jeux × 7 settings = 35 rows. Setting 4 (MAPS) bold.

### 3a. Seaquest

| Setting      | 2nd-Net | Cascade | Training R.   | Train Z           | Val R.        | Val Z             |
|--------------|:-------:|:-------:|:-------------:|:-----------------:|:-------------:|:-----------------:|
| 1            | No      | No      | 1.48 ± 0.29   | —                 | 1.21 ± 0.16   | —                 |
| 2            | No      | 1st     | 0.90 ± 0.21   | **-2.32 (Yes)**   | 0.76 ± 0.19   | **-2.59 (Yes)**   |
| 3            | Yes     | No      | 1.04 ± 0.57   | -0.97 (No)        | 0.97 ± 0.61   | -0.53 (No)        |
| **4 MAPS**   | Yes     | 1st     | **3.71 ± 0.40** | **6.46 (Yes)**  | **3.06 ± 0.34**| **7.03 (Yes)**   |
| 5            | Yes     | 2nd     | 4.10 ± 0.86   | **1.97 (Yes)**    | 3.99 ± 1.84   | **2.14 (Yes)**    |
| 6            | Yes     | Both    | 5.56 ± 1.85   | **3.09 (Yes)**    | 6.15 ± 2.34   | **2.98 (Yes)**    |
| 7 ACB        | No      | No      | 0.81 ± 0.03   | **-3.26 (Yes)**   | 0.63 ± 0.26   | **-2.65 (Yes)**   |

### 3b. Asterix

| Setting      | 2nd-Net | Cascade | Training R.   | Train Z           | Val R.        | Val Z             |
|--------------|:-------:|:-------:|:-------------:|:-----------------:|:-------------:|:-----------------:|
| 1            | No      | No      | 3.49 ± 3.32   | —                 | 2.49 ± 1.94   | —                 |
| 2            | No      | 1st     | 1.38 ± 0.67   | -0.88 (No)        | 1.59 ± 0.90   | -0.60 (No)        |
| 3            | Yes     | No      | 6.27 ± 1.87   | 1.03 (No)         | 5.48 ± 1.30   | 1.81 (No)         |
| **4 MAPS**   | Yes     | 1st     | 4.95 ± 1.05   | 0.59 (No)         | 4.54 ± 1.01   | 1.32 (No)         |
| 5            | Yes     | 2nd     | 3.40 ± 0.30   | -0.04 (No)        | 2.94 ± 0.25   | 0.32 (No)         |
| 6            | Yes     | Both    | 5.42 ± 0.69   | 0.81 (No)         | 5.77 ± 0.94   | **2.15 (Yes)**    |
| 7 ACB        | No      | No      | 2.75 ± 0.09   | -0.31 (No)        | 2.13 ± 1.07   | -0.23 (No)        |

### 3c. Breakout

| Setting      | 2nd-Net | Cascade | Training R.   | Train Z           | Val R.        | Val Z             |
|--------------|:-------:|:-------:|:-------------:|:-----------------:|:-------------:|:-----------------:|
| 1            | No      | No      | 5.96 ± 0.43   | —                 | 6.10 ± 0.21   | —                 |
| 2            | No      | 1st     | 7.08 ± 0.65   | **2.05 (Yes)**    | 6.84 ± 0.41   | **2.25 (Yes)**    |
| 3            | Yes     | No      | 6.12 ± 0.51   | 0.36 (No)         | 6.22 ± 0.69   | 0.24 (No)         |
| **4 MAPS**   | Yes     | 1st     | **8.54 ± 0.07** | **8.36 (Yes)**  | **8.07 ± 0.72**| **3.70 (Yes)**   |
| 5            | Yes     | 2nd     | 6.23 ± 0.14   | 0.85 (No)         | 6.16 ± 0.05   | 0.39 (No)         |
| 6            | Yes     | Both    | 8.08 ± 0.24   | **6.06 (Yes)**    | 7.91 ± 1.36   | 1.87 (No)         |
| 7 ACB        | No      | No      | 12.36 ± 1.13  | **7.49 (Yes)**    | 11.67 ± 2.70  | **2.90 (Yes)**    |

### 3d. Space Invaders

| Setting      | 2nd-Net | Cascade | Training R.    | Train Z           | Val R.         | Val Z             |
|--------------|:-------:|:-------:|:--------------:|:-----------------:|:--------------:|:-----------------:|
| 1            | No      | No      | 23.62 ± 0.86   | —                 | 19.71 ± 1.84   | —                 |
| 2            | No      | 1st     | 31.52 ± 1.01   | **8.46 (Yes)**    | 29.62 ± 2.28   | **4.79 (Yes)**    |
| 3            | Yes     | No      | 21.14 ± 2.41   | -1.37 (No)        | 20.95 ± 2.60   | 0.55 (No)         |
| **4 MAPS**   | Yes     | 1st     | **26.84 ± 0.11** | **5.28 (Yes)**  | **26.80 ± 1.59**| **4.13 (Yes)**   |
| 5            | Yes     | 2nd     | 28.11 ± 0.24   | **7.14 (Yes)**    | 27.95 ± 1.10   | **5.45 (Yes)**    |
| 6            | Yes     | Both    | 26.57 ± 0.72   | **3.74 (Yes)**    | 22.97 ± 0.89   | **2.26 (Yes)**    |
| 7 ACB        | No      | No      | 71.50 ± 17.38  | **3.89 (Yes)**    | 59.26 ± 16.61  | **3.34 (Yes)**    |

### 3e. Freeway

| Setting      | 2nd-Net | Cascade | Training R.    | Train Z           | Val R.         | Val Z             |
|--------------|:-------:|:-------:|:--------------:|:-----------------:|:--------------:|:-----------------:|
| 1            | No      | No      | 30.96 ± 5.35   | —                 | 29.03 ± 7.12   | —                 |
| 2            | No      | 1st     | 32.77 ± 8.23   | 0.26 (No)         | 32.57 ± 8.82   | 0.44 (No)         |
| 3            | Yes     | No      | 24.72 ± 17.58  | -0.48 (No)        | 25.00 ± 17.68  | -0.30 (No)        |
| **4 MAPS**   | Yes     | 1st     | 35.53 ± 0.08   | 1.21 (No)         | 34.20 ± 2.83   | 0.95 (No)         |
| 5            | Yes     | 2nd     | 39.41 ± 1.89   | **2.11 (Yes)**    | 34.40 ± 2.97   | 0.98 (No)         |
| 6            | Yes     | Both    | 33.84 ± 2.03   | 0.71 (No)         | 30.63 ± 2.36   | 0.30 (No)         |
| 7 ACB        | No      | No      | 0.19 ± 0.10    | **-8.14 (Yes)**   | 0.17 ± 0.05    | **-5.73 (Yes)**   |

**SARL headline targets à reproduire à ±2σ — Setting 4 MAPS validation Z :**
- Seaquest **7.03** (strongest).
- Breakout **3.70**.
- Space Invaders **4.13**.

**"Edge of noise" envs** (Val Z < 2 Setting 4) : Asterix, Freeway. **Ne pas chercher à les battre**
artificiellement ; paper lui-même ne les revendique pas comme significatifs.

---

## 4. SARL + Continual Learning

Paper §CL Results p.17-18 évalue la **rétention** (fraction des rewards prior-task préservée après
training sur la tâche suivante) plutôt que des z-scores absolus. Curriculum :
**Breakout → Space Invaders → Seaquest → Freeway**. 100 000 steps par env.

| Métrique                                         | Paper value              |
|--------------------------------------------------|:------------------------:|
| Mean retention (across settings, all envs)       | **45.1 % ± 31.1 %**      |
| Best case retention (settings w/ low-weight reg) | **up to 84.2 %**         |
| Baseline DQN retention (no MAPS)                 | ≈ random policy (≈ 0 %)  |

**Headline target** : MAPS conserve > 50 % prior-task return sur Breakout après training séquentiel
sur les 4 envs — paper p.18 Figure 9 top panel.

Paper ne reporte pas de z-scores settings-par-settings sur CL. L'exploration est "MAPS vs DQN
baseline" sur les courbes de retention.

---

## 5. MARL — paper Table 7 (MeltingPot 2.0)

4 substrates × 7 settings = 28 rows. **Limitation paper-declared** :
`D-marl-cascade-not-implemented` → settings 2/4/5/6 MARL ne sont pas paper-faithful côté cascade.

### 5a. Commons Harvest Partnership

| Setting      | 2nd-Net | Cascade | Training R.    | Train Z          | Dist Entropy  | Dist Z           |
|--------------|:-------:|:-------:|:--------------:|:----------------:|:-------------:|:----------------:|
| 1            | No      | No      | 29.21 ± 0.72   | —                | 1.82 ± 0.07   | —                |
| 2            | No      | 1st     | 27.15 ± 0.04   | **-4.07 (Yes)**  | 1.71 ± 0.06   | -1.73 (No)       |
| 3            | Yes     | No      | 29.03 ± 0.29   | -0.34 (No)       | 1.88 ± 0.03   | 1.04 (No)        |
| **4 MAPS**   | Yes     | 1st     | **34.52 ± 0.98** | **6.20 (Yes)** | 1.64 ± 0.10   | **-2.20 (Yes)**  |
| 5            | Yes     | 2nd     | 28.63 ± 0.68   | -1.03 (No)       | 1.88 ± 0.05   | 0.93 (No)        |
| 6            | Yes     | Both    | 33.01 ± 2.57   | **2.01 (Yes)**   | 1.62 ± 0.11   | **-2.28 (Yes)**  |
| 7 ACB        | No      | No      | 24.58 ± 10.56  | -0.75 (No)       | NA            | NA               |

### 5b. Commons Harvest Closed

| Setting      | 2nd-Net | Cascade | Training R.    | Train Z          | Dist Entropy  | Dist Z            |
|--------------|:-------:|:-------:|:--------------:|:----------------:|:-------------:|:-----------------:|
| 1            | No      | No      | 19.52 ± 0.71   | —                | 1.89 ± 0.01   | —                 |
| 2            | No      | 1st     | 26.05 ± 0.71   | **9.18 (Yes)**   | 1.72 ± 0.01   | **-20.93 (Yes)**  |
| 3            | Yes     | No      | 19.15 ± 0.42   | -0.64 (No)       | 1.86 ± 0.01   | **-4.51 (Yes)**   |
| **4 MAPS**   | Yes     | 1st     | **25.21 ± 1.06** | **6.31 (Yes)** | 1.76 ± 0.12   | -1.64 (No)        |
| 5            | Yes     | 2nd     | 18.81 ± 0.68   | -1.02 (No)       | 1.86 ± 0.05   | -0.86 (No)        |
| 6            | Yes     | Both    | 23.97 ± 1.30   | **4.25 (Yes)**   | 1.77 ± 0.01   | **-15.96 (Yes)**  |
| 7 ACB        | No      | No      | 39.39 ± 10.69  | **3.19 (Yes)**   | NA            | NA                |

### 5c. Chemistry — Three Metabolic Cycles w/ Plentiful Distractors

| Setting      | 2nd-Net | Cascade | Training R.    | Train Z          | Dist Entropy  | Dist Z            |
|--------------|:-------:|:-------:|:--------------:|:----------------:|:-------------:|:-----------------:|
| 1            | No      | No      | 1.18 ± 0.09    | —                | 1.95 ± 0.01   | —                 |
| 2            | No      | 1st     | 1.13 ± 0.01    | -0.70 (No)       | 1.81 ± 0.05   | **-3.61 (Yes)**   |
| 3            | Yes     | No      | 1.25 ± 0.01    | 1.07 (No)        | 1.93 ± 0.02   | -1.15 (No)        |
| **4 MAPS**   | Yes     | 1st     | 1.11 ± 0.01    | -0.91 (No)       | 1.83 ± 0.12   | -1.33 (No)        |
| 5            | Yes     | 2nd     | 1.23 ± 0.06    | 0.72 (No)        | 1.90 ± 0.01   | **-3.66 (Yes)**   |
| 6            | Yes     | Both    | 1.15 ± 0.03    | -0.37 (No)       | 1.81 ± 0.19   | -1.03 (No)        |
| 7 ACB        | No      | No      | 1.41 ± 0.60    | 0.64 (No)        | NA            | NA                |

### 5d. Territory Inside Out

| Setting      | 2nd-Net | Cascade | Training R.     | Train Z           | Dist Entropy  | Dist Z            |
|--------------|:-------:|:-------:|:---------------:|:-----------------:|:-------------:|:-----------------:|
| 1            | No      | No      | 57.94 ± 6.84    | —                 | 1.98 ± 0.00   | —                 |
| 2            | No      | 1st     | 41.98 ± 7.43    | **-2.24 (Yes)**   | 2.01 ± 0.25   | 0.18 (No)         |
| 3            | Yes     | No      | 54.04 ± 0.34    | -0.81 (No)        | 1.97 ± 0.02   | -0.66 (No)        |
| **4 MAPS**   | Yes     | 1st     | 48.47 ± 1.45    | -1.92 (No)        | 1.58 ± 0.08   | **-7.16 (Yes)**   |
| 5            | Yes     | 2nd     | 58.83 ± 6.15    | 0.14 (No)         | 1.92 ± 0.07   | -1.29 (No)        |
| 6            | Yes     | Both    | 46.05 ± 9.16    | -1.47 (No)        | 1.68 ± 0.12   | **-3.46 (Yes)**   |
| 7 ACB        | No      | No      | 101.99 ± 38.67  | 1.89 (No)         | NA            | NA                |

**MARL headline — "MAPS wins" cases** :
- Harvest Closed Setting 2 Train Z = **9.18**, Setting 4 Train Z = **6.31**.
- Harvest Partnership Setting 4 Train Z = **6.20**.

**"MAPS does NOT beat baseline"** (paper admits §5 p.20) : Chemistry, Territory Inside Out. Report
honnêtement, pas de tuning.

---

## Phase F reproduction run budget

Référence : `docs/plans/plan-20260419-review-and-reproduce.md` §Phase F.

| Domain          | Cells total                                      | Per-cell estimate                    | Hardware          |
|-----------------|--------------------------------------------------|---------------------------------------|-------------------|
| Blindsight      | 500 × 6 = **3 000**                              | ~minutes (200 epochs, 100-dim)        | Mac CPU / local   |
| AGL             | 500 × 6 × 2 awareness = **6 000**                | ~minutes                              | Mac CPU / local   |
| SARL            | 5 games × 7 settings × 3 seeds = **105**         | 500k-1M frames ; 5-15 h cascade       | Tamia CPU / GPU   |
| SARL+CL         | 4 envs × 7 settings × 3 seeds = **84**           | 100k steps × 4 envs ; qq h            | Tamia GPU         |
| MARL            | 4 substrates × 7 settings × 3 seeds = **84**     | 300k env steps ; ~16 h / seed A100    | Tamia H100        |
| **Total**       | **9 273 cells**                                  | wall-time dominée par MARL            |                   |

**Compute discipline (Sprint-08 Phase A.4)** : ≤ 2 jobs concurrents sur `aip-gdumas85`, chaînés via
`scripts/slurm/submit_chained.sh --dependency=afterany`.

---

## Current reproduction status — Sprint-06 gaps

| Gap    | Domain     | Paper headline               | Our N=10 (Sprint-06)     | Status                                                          |
|--------|------------|------------------------------|--------------------------|-----------------------------------------------------------------|
| RG-001 | AGL eval   | Eval path ported             | —                        | ✅ resolved in Sprint-06                                         |
| RG-002 | Blindsight | 0.97 / z=9.01                | 0.755 / z=+0.40          | 🔴 → D-blindsight-hidden-dim + D-blindsight-metric-mismatch (D.25) |
| RG-003 | AGL        | 0.66 high / 0.62 low         | ≈0.07-0.09 / z≈0         | 🔴 → D-agl-training-missing (D.28 port phase 2)                  |

Voir `docs/reproduction/deviations.md` pour la liste complète des **52 issues trackées**.

---

## Known risks

1. **Seed control** — code student original n'avait pas de seeding global ; notre port le fixe,
   mais les z-scores papier proviennent de runs à "seed meaning" flou. Nous reproduisons la
   **méthode**, pas des chiffres bit-exacts. Écarts > ±2σ ne sont pas nécessairement des échecs.
2. **Cascade α=0.02, n_iter=50** — "empirically selected" (paper §2.1). Sensibilité à tester en
   follow-up si budget.
3. **MinAtar version drift** — student a vendored `external/MinAtar/` ; on pin le SHA dans les
   tests parity.
4. **MeltingPot version** — les substrate rewards ont changé entre 2.0 et 2.1 ; pin commit en
   Phase E.8 (recipe dans `docs/install_meltingpot.md` à créer).
5. **🆘 Paper-vs-student divergence** — 15 cas documentés dans `deviations.md` : le code student
   shipped dans `external/paper_reference/` **ne peut pas** avoir produit les Tables 5/6/7 papier
   (divergences hardcodées sur `GAMMA`, `lr_second_order`, `alpha_ema`, Adam betas, scheduler
   step, etc.). **Nos runs alignés sur paper Tables 9/10/11/12 vont produire des chiffres
   différents de ce que le student a publié**. À documenter honnêtement en Phase G.

---

## Change log

- **2026-04-19 (B.13)** : rewrite complet — seeds corrigés (paper 500 perceptual / 3 RL, pas 10) ;
  7 settings incluant ACB ; tables par-domaine étendues à 7×N_env rows ; liens vers
  `paper_targets_extracted.md` + `deviations.md` ; Phase F budget revu ; ajout risque
  paper-vs-student.
- **2026-04-18 (Sprint-06)** : matrix initiale avec valeurs mesurées Sprint-06.
