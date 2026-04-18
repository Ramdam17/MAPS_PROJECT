# Paper ↔ code deviations

This file logs every point where the MAPS TMLR paper description and the
reference implementations (`Blindsight_TMLR.py`, `AGL_TMLR.py`, `SARL_CL/.../maps.py`)
disagree. The reference **code** is the parity target for now — deviating from
it breaks bit-for-bit reproducibility of Juan's runs. Deviations toward the
paper are logged here and exposed as configurable knobs so future work can
A/B them without forking.

| # | Location | Paper says | Code does | Chosen default | Exposed as |
|---|----------|-----------|-----------|---------------|-----------|
| D-001 | Wagering head | 2 units with softmax over {bet, no-bet} (eq.2-3, Koch & Preuschoff 2007) | 1 unit with sigmoid → scalar confidence in [0, 1] | `n_wager_units=1` (parity) | `WageringHead(n_wager_units=...)` / `SecondOrderNetwork(n_wager_units=...)` |
| D-002 | First-order loss | "Contrastive loss" (eq.4, Chen et al. 2020 SimCLR form) | Contractive AutoEncoder loss (Rifai et al. 2011) — both are sometimes called "contrastive" in older literature | `cae_loss` (parity) | `maps.components.losses.cae_loss` (SimCLR variant not yet implemented) |
| D-003 | Distillation KL scaling | Hinton (2015) recommends multiplying soft loss by T² | Reference code does **not** scale by T² | unscaled (parity) | doc note in `distillation_loss`; add `scale_by_t_squared=True` later if needed |
| D-004 | AGL decoder activation | Global sigmoid on output vector | Sigmoid applied independently on each 6-bit letter chunk (AGL only, not Blindsight) | matches reference, AGL-specific | `FirstOrderMLP(decoder_activation=make_chunked_sigmoid(6))` |

## Policy

- When a paper-faithful variant is implemented, it is **not** the default. The
  default matches the reference code so that `pytest tests/parity` stays green.
- Each row above must remain truthful. If you change a default, update this
  table and the relevant docstring in the same commit.
- When running ablations that toggle these knobs, record which ones in the
  experiment YAML (`config/experiments/*.yaml`) so downstream analyses can
  filter.

## Sprint 06 reproduction gaps (2026-04-18)

None of the entries below are deviations from the reference code — they are
missing pieces of the paper-faithful reproduction pipeline that were not
recovered during the Sprint 04b monolith deletion. They belong here as "what
the current MAPS core does *not* reproduce yet" and are tracked in `docs/TODO.md`
under "Reproduction gaps".

| # | Surface | Paper headline | Our number (N=10) | Root cause | Tracking ID |
|---|---------|---------------:|------------------:|------------|:-----------:|
| G-01 | Blindsight detection acc. (full MAPS) | 0.97 (z=9.01) | 0.755 discrim. / 0.71 wager (z=+0.40) | Metric-definition mismatch candidate: paper "detection accuracy" may refer to wager-head binary classification, an average over conditions, or a different threshold protocol. Eval code path is ported and correct per reference `testing()`. | RG-002 |
| G-02 | AGL High Awareness classification acc. | 0.66 (z=8.20) | 0.073 (z=+0.00) | `AGLTrainer.pre_train` resets the first-order to init weights (reference L751 behavior). Paper numbers come from a downstream supervised phase on Grammar A vs B that used the *pre-trained* second-order + a fresh first-order — this phase is not ported. | RG-003 |
| G-03 | AGL Low Awareness classification acc. | 0.62 (z=15.70) | 0.093 (z=+0.00) | Same root cause as G-02 — the post-hoc seed-pool split cannot create the "awareness" signal without the downstream training phase. | RG-003 |

No hyperparameters were changed from `config/training/{blindsight,agl}.yaml` in
Sprint 06. The gaps above are reproduction-depth, not protocol drift.
