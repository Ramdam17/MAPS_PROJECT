# Phase δ — SARL Setting 7 (ACB) vs paper Table 6

Reproduction of Young & Tian 2019 AC(λ) baseline on 5 MinAtar games × 3 seeds.
Headline metric : Training Rewards = `avg_return_ema` (0.99-EMA of episode returns).
Validation : `last_validation_mean` (mean of 2 final eval episodes).

| Game | Metric | Ours (3 seeds) | Paper | ±2σ | Δ |
|---|---|---|---|:---:|---:|
| seaquest | train | 0.654 ± 0.078 | 0.81 ± 0.03 | ✗ | -5.20σ |
| seaquest | val   | 1.167 ± 0.764 | 0.63 ± 0.26 | ✗ | +2.06σ |
| asterix | train | 0.709 ± 0.173 | 2.75 ± 0.09 | ✗ | -22.68σ |
| asterix | val   | 0.333 ± 0.577 | 2.13 ± 1.07 | ✓ | -1.68σ |
| breakout | train | 6.680 ± 1.574 | 12.36 ± 1.13 | ✗ | -5.03σ |
| breakout | val   | 6.667 ± 2.021 | 11.67 ± 2.70 | ✓ | -1.85σ |
| space_invaders | train | 32.152 ± 7.175 | 71.50 ± 17.38 | ✗ | -2.26σ |
| space_invaders | val   | 34.333 ± 30.892 | 59.26 ± 16.61 | ✓ | -1.50σ |
| freeway | train | 0.119 ± 0.037 | 0.19 ± 0.10 | ✓ | -0.71σ |
| freeway | val   | 0.167 ± 0.289 | 0.17 ± 0.05 | ✓ | -0.07σ |

**Method note** : ACB is structurally distinct from settings 1-6 — online AC(λ) with
eligibility traces, custom debiased RMSprop, no replay buffer. Port at
`src/maps/experiments/sarl/actor_critic.py` is bit-identical to vendored
`external/paper_reference/sarl_ac_lambda.py` (parity test atol=0 over 30 update
steps × 3 seeds). 500_000 frames per cell, ~5-15 min on Tamia CPU.

**Freeway note** : the original 2026-05-19 run hit `n_validation_points=0` due to
a `setdefault` masking bug (10eb8bb). Re-ran 3 seeds with the fix in place ;
validation now populated.