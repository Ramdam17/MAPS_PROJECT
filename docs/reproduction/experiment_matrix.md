# Reproduction Target Matrix

**Source:** MAPS TMLR submission (Vargas et al., 2025) — `pdf/MAPS_TMLR_Journal_Submission.pdf` and README.md Results section.

**Comparison baseline:** second-order OFF + cascade OFF (standard network).
**Z-score:** computed vs. baseline mean/std over seeds (paper N=10 seeds).

Our reproduction tolerance: **±2σ** around reported mean.

---

## 1. Perceptual Tasks

| Task | Metric | Paper mean | Paper std | Z-score vs baseline | Our target range (±2σ) |
|------|--------|-----------:|----------:|--------------------:|-----------------------:|
| Blindsight | detection acc. | 0.97 | 0.02 | 9.01 | [0.93, 1.01] |
| AGL — High Awareness | classification acc. | 0.66 | 0.05 | 8.20 | [0.56, 0.76] |
| AGL — Low Awareness | classification acc. | 0.62 | 0.07 | 15.70 | [0.48, 0.76] |

## 2. SARL — MinAtar DQN

| Env | Metric | Paper mean | Paper std | Z-score | Our target range |
|-----|--------|-----------:|----------:|--------:|-----------------:|
| Seaquest | mean return (final 100 eps) | 3.06 | 0.34 | 7.03 | [2.38, 3.74] |
| Asterix | mean return | 4.54 | 1.01 | 1.32 | [2.52, 6.56] |
| Breakout | mean return | 8.07 | 0.72 | 3.70 | [6.63, 9.51] |
| Space Invaders | mean return | 26.80 | 1.59 | 4.13 | [23.62, 29.98] |
| Freeway | mean return | 34.20 | 2.83 | 0.95 | [28.54, 39.86] |

**Note:** Asterix and Freeway z-scores are low (<2) — reproducing within tolerance is already "at the edge of noise". Treat as exploratory.

## 3. SARL + Continual Learning

- Transfer learning: mean retention **45.1% ± 31.1%** (σ is very large — low confidence)
- Continual learning: paper labels as "exploratory" — no point target

## 4. MARL — MeltingPot 2.0 MAPPO

| Substrate | Metric | Paper mean | Paper std | Z-score | Our target range |
|-----------|--------|-----------:|----------:|--------:|-----------------:|
| Commons Harvest Partnership | mean return | 34.52 | 0.98 | 6.20 | [32.56, 36.48] |
| Commons Harvest Closed | mean return | 25.21 | 1.06 | 6.31 | [23.09, 27.33] |
| Chemistry (3 Metabolic) | mean return | 1.11 | 0.05 | **-0.91** | [1.01, 1.21] |
| Territory Inside Out | mean return | 48.47 | 1.45 | **-1.92** | [45.57, 51.37] |

**Note:** Chemistry and Territory have negative z-scores — MAPS does NOT outperform baseline here. Paper is honest about this.

---

## Experimental Factors (2×2×... design)

Each experiment is run under 6 settings combining:
- **Second-order network:** ON / OFF
- **Cascade model:** ON / OFF
- (+2 extra settings per paper — TBD from code inspection of the `setting` branches in `Blindsight_TMLR.py:1804-1898`)

Settings 1-6 mapping (from Blindsight code):

| Setting | 2nd-order | Cascade | Notes |
|---------|-----------|---------|-------|
| 1 | OFF | OFF | baseline |
| 2 | ON | OFF | MAPS — second-order only |
| 3 | OFF | ON | MAPS — cascade only |
| 4 | ON | ON | full MAPS |
| 5 | TBD | TBD | (inspect code) |
| 6 | TBD | TBD | (inspect code) |

---

## Seeds

- **Paper:** N=10 seeds per cell
- **Our reproduction target:**
  - Blindsight/AGL: 10 seeds (cheap)
  - SARL: 5 seeds × 5 envs × 6 settings = 150 runs
  - MARL: 3 seeds × 4 substrates × 6 settings = 72 runs
- **Budget check:** ~250 runs total ≥ substantial Narval allocation. Prioritize Blindsight + AGL + 2 SARL envs first.

---

## Known Risks

1. **Seed control was absent** in original code — "paper seeds 1-10" may not reproduce identical results even with the original git SHA. We reproduce the **method**, not the exact numbers.
2. **Cascade α=0.02 / iter=50** was "empirically selected" (paper §2.1). Sensitivity analysis is a candidate sprint 01 task.
3. **MinAtar version drift** — check which tag of `kenjyoung/MinAtar` was used.
4. **MeltingPot version** — substrate rewards have changed between 2.0 and 2.1; pin version.
