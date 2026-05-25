# Paper reference source — READ ONLY

This directory contains the **original student code** (Juan David Vargas,
Zahra Sheikhbahaee) from before Sprint 04b's legacy delete. It is the code
that produced the numbers in the MAPS TMLR submission paper (Vargas et al.).

**Do not edit.** These files are here as ground-truth references to compare
against the ports in `src/maps/` and to serve as a fallback when the papers'
Appendix B hyperparameter tables are ambiguous.

| File | Restored from | Size | Used by |
|---|---|---:|---|
| `sarl_maps.py` | `external/MinAtar/examples/maps.py` @ `d90d8f8^` | 2 721 L | Source for `src/maps/experiments/sarl/` |
| `sarl_cl_maps.py` | `SARL_CL/examples_cl/maps.py` @ `d90d8f8^` | 2 580 L | Source for `src/maps/experiments/sarl_cl/` |
| `agl_tmlr.py` | `AGL/AGL_TMLR.py` @ `d90d8f8^` | 2 785 L | Source for `src/maps/experiments/agl/` |
| `blindsight_tmlr.py` | `BLINDSIGHT/Blindsight_TMLR.py` @ `d90d8f8^` | 2 419 L | Source for `src/maps/experiments/blindsight/` |
| `sarl_ac_lambda.py` | `external/MinAtar/examples/AC_lambda.py` | 498 L | Paper Setting 7 (ACB) baseline — not yet ported |
| `marl_tmlr/` (dir, 188 files ~2 MB) | `MARL/MAPPO-ATTENTION/` (forked from `neuronphysics/MAPPO-ATTENTION`, MAPPO Yu et al. 2022) | ~30k L | Source for `src/maps/experiments/marl/` (Phase E port ; restored at `db3e2af`, moved here at E.1) |

## Canonical reproduction target

The **paper PDF** (`pdf/MAPS_TMLR_Journal_Submission.pdf`) is the ultimate
source of truth for hyperparameters and methodology. When student code and
paper disagree, the paper wins (décision Rémy 2026-04-19). Known divergences
are logged in `docs/reproduction/deviations.md`.

## History

- 2026-04-17 (Sprint 04b commit `d90d8f8`): the 4 monoliths (`sarl_maps.py`,
  `sarl_cl_maps.py`, `agl_tmlr.py`, `blindsight_tmlr.py`) were ported into
  `src/maps/experiments/` and deleted from the working tree. Parity tests in
  `tests/parity/` copied specific functions to `_reference_*.py` slices.
- 2026-04-19 (Sprint 07 audit): the full monoliths were restored here. Slices
  remain in `tests/parity/` for atol=1e-6 Tier 1/2/3 testing; full monoliths
  sit here for design/hyperparameter auditing. Rémy's mandate: *preserve
  everything that reproduces the paper*.
