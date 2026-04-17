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
