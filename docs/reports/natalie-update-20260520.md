# Update for N. Kastel — Sprint 08 D.31 closeout (2026-05-20)

> Draft of the message to send. Not auto-emailed — copy/paste into Slack or email when ready.

---

Hey Natalie,

Two updates after your Know-Thyself feedback.

## 1. You were right — Settings 4 and 5 were missing

You correctly flagged that the May 1 archive only ships 4 cells per Know-Thyself
domain (`neither`, `cascade_only`, `second_order_only`, `both`), not 6. The diagnosis
is a bit deeper than "we didn't run them yet":

- Our legacy `BlindsightTrainer` / `AGLTrainer` applied cascade **symmetrically**
  across both networks. As a result the cell we labelled `both` was producing paper
  **Setting 6** numbers (cascade on both nets), not the headline **Setting 4 (MAPS)**
  (cascade on 1st net only). The earlier D.25 closeout comparison
  (0.94 / 0.82) was actually validating Setting 6 against Setting 4 numbers — they
  pass by coincidence because Table 5a Settings 4 and 6 are numerically close.
- We refactored `BlindsightSetting` / `AGLSetting` to expose
  `(cascade_1st, cascade_2nd, second_order)`, mirroring `MarlSetting`. Parity
  preserved at atol=1e-5 on the 4 legacy cells (so the May 1 archive is still
  consistent with the new code).
- We then ran the actual Settings 4 and 5 over 500 seeds each (2000 new runs total).

Phase γ result vs paper Table 5a/b/c — **all 12 metrics within ±2σ** :

| | Setting 4 (MAPS) ours | Paper S4 | Setting 5 ours | Paper S5 |
|---|---|---|---|---|
| Blindsight disc (super) | 0.937 ± 0.034 | 0.97 ± 0.02 | 0.918 ± 0.037 | 0.96 ± 0.03 |
| Blindsight wager (super) | 0.800 ± 0.045 | 0.85 ± 0.04 | 0.815 ± 0.043 | 0.87 ± 0.04 |
| AGL high prec | 0.649 ± 0.028 | 0.66 ± 0.05 | 0.625 ± 0.027 | 0.63 ± 0.04 |
| AGL high wager | 0.591 ± 0.032 | 0.58 ± 0.06 | 0.612 ± 0.031 | 0.61 ± 0.06 |
| AGL low prec | 0.615 ± 0.049 | 0.62 ± 0.07 | 0.548 ± 0.054 | 0.56 ± 0.07 |
| AGL low wager | 0.833 ± 0.046 | 0.82 ± 0.07 | 0.856 ± 0.045 | 0.87 ± 0.07 |

Blindsight residuals -1.2 to -1.7σ — consistent with the D.25 finding
*"within seed noise of paper std"*. AGL residuals all < 0.25σ.

New bundle attached as `knowyourself_outputs_20260520.tar.gz` (261 MB, 6 cells × 500
seeds × 2 experiments). Same dir layout as the May 1 archive plus
`setting-4-maps/` and `setting-5-cascade-2nd/`.

## 2. SARL counts — I got it wrong too

You asked about the SARL counts and I told you "5 games × 6 settings × 2 variants".
That was wrong on two points :

- **7 settings, not 6.** The paper adds **Setting 7 (ACB)** — Actor-Critic Baseline
  from Young & Tian 2019 (AC(λ), λ=0.8). It's a structurally different algorithm
  from settings 1-6 (online actor-critic with eligibility traces, no replay buffer,
  custom debiased RMSprop). We've now ported it (`src/maps/experiments/sarl/actor_critic.py`)
  with bit-identical parity to Young & Tian's reference, and ran 5 games × 3 seeds = 15 cells.
  Bundle attached as `sarl_setting7_acb_outputs_20260520.tar.gz` (7.3 MB).
- **SARL+CL is a chained curriculum, not parallel.** It's not the same 5 games as
  SARL — it's specifically Breakout → SpaceInvaders → Seaquest → Freeway in that
  order, each stage initialized from the previous teacher checkpoint
  (paper §A.3). So 1 curriculum × 7 settings × 3 seeds = 21 runs spawning 84 stage rows.

Honest note on the SARL ACB reproduction : training rewards diverge from paper
Table 6 on 3 of 5 games (asterix, breakout, seaquest training R are 2–22σ below
paper). Validation rewards mostly within 2σ. Likely cause is a budget mismatch —
paper Table 11 says 500k frames, but main text page 13 says "1 million steps"
(flagged as `D-sarl-num-frames` in our deviations log). We ran at 500k. If
you need the paper-text-aligned numbers, I can re-run at 1M and update.

## 3. MARL Setting 7 — still open

The paper's MARL ACB is **not** the same algorithm as SARL ACB. Per the Agapiou
2023 Melting Pot paper (which MAPS cites for the MARL baseline), it's
**IMPALA + CPC auxiliary loss** on LSTM state representations (Espeholt 2018 +
Oord 2018). DeepMind's public meltingpot repo only ships the substrates plus an
RLlib PPO example, not the ACB training code. I checked all 18 of Juan's public
GitHub repos — no `setting-7` or `IMPALA` code path is present anywhere.

Guillaume's Slack thread with Juan should clarify which code produced the MARL
Table 7 Setting-7 row. Until then, MARL Setting 7 stays open.

---

Attachments :
- `knowyourself_outputs_20260520.tar.gz` (261 MB)
- `sarl_setting7_acb_outputs_20260520.tar.gz` (7.3 MB)

Full reports (statistics + paper deltas in σ units) live at :
- `docs/reports/phase-gamma-settings-4-5.md`
- `docs/reports/phase-delta-acb-sarl.md`

Code changes (4 commits, branch `repro/paper-faithful`) :
- `7f6aee7` refactor(know-thyself) : asymmetric cascade dataclasses
- `6e5ef76` feat(know-thyself) : factorial_6cell YAML + CLI plumbing
- `263a437` build(slurm) : Settings 4/5 production sbatch
- `e75ffb3` docs(repro) : Phase γ aggregate (all 12 metrics ±2σ)
- `3df5e33` feat(sarl) : AC(λ) port as Setting 7 (ACB)
- `10eb8bb` fix(sarl) : freeway validation cadence regression
- `c8c2652` docs(repro) : D.31 closeout doc updates

Happy to walk through any of this on a call if useful.

Rémy
