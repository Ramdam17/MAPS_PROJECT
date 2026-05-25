# Phase γ — Blindsight & AGL Settings 4 and 5 (500 seeds each)

Reproduction comparison against paper Table 5a/5b/5c. ✓ = within ±2σ of paper.


## Blindsight (paper Table 5a — superthreshold)

| Setting | Metric | Our (500 seeds) | Paper | ±2σ | Δ |
|---|---|---|---|:---:|---:|
| setting-4-maps | disc | 0.937 ± 0.034 | 0.97 ± 0.02 | ✓ | -1.65σ |
| setting-4-maps | wager | 0.800 ± 0.045 | 0.85 ± 0.04 | ✓ | -1.24σ |
| setting-5-cascade-2nd | disc | 0.918 ± 0.037 | 0.96 ± 0.03 | ✓ | -1.39σ |
| setting-5-cascade-2nd | wager | 0.815 ± 0.043 | 0.87 ± 0.04 | ✓ | -1.36σ |

## AGL (paper Table 5b high + Table 5c low awareness)

| Setting | Metric | Our (500 seeds) | Paper | ±2σ | Δ |
|---|---|---|---|:---:|---:|
| setting-4-maps | high.prec_1st | 0.649 ± 0.028 | 0.66 ± 0.05 | ✓ | -0.21σ |
| setting-4-maps | high.wager | 0.591 ± 0.032 | 0.58 ± 0.06 | ✓ | +0.18σ |
| setting-4-maps | low.prec_1st | 0.615 ± 0.049 | 0.62 ± 0.07 | ✓ | -0.07σ |
| setting-4-maps | low.wager | 0.833 ± 0.046 | 0.82 ± 0.07 | ✓ | +0.19σ |
| setting-5-cascade-2nd | high.prec_1st | 0.625 ± 0.027 | 0.63 ± 0.04 | ✓ | -0.13σ |
| setting-5-cascade-2nd | high.wager | 0.612 ± 0.031 | 0.61 ± 0.06 | ✓ | +0.04σ |
| setting-5-cascade-2nd | low.prec_1st | 0.548 ± 0.054 | 0.56 ± 0.07 | ✓ | -0.17σ |
| setting-5-cascade-2nd | low.wager | 0.856 ± 0.045 | 0.87 ± 0.07 | ✓ | -0.21σ |
