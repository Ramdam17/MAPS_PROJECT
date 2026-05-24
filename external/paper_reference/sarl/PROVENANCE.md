# Juan's canonical SARL reference (restored)

This directory holds the **canonical** SARL source code as published by Juan
David Vargas on his fork's `main` branch, and the shell launcher that invokes
it. These files are the **ground truth** for the SARL numbers reported in the
MAPS TMLR submission (Table 6).

## Files

| File | Provenance |
|------|------------|
| `maps_v1.py` | Restored from `git show 8aa1138^:SARL/MinAtar/examples/maps_v1.py` on 2026-05-24. Also live at <https://github.com/juandavidvargas19/MAPS_PROJECT/blob/main/SARL/MinAtar/examples/maps_v1.py>. |
| `SARL_Training_Standard.sh` | Restored from `git show 8aa1138^:SARL/SARL_Training_Standard.sh` on 2026-05-24. Also live at <https://github.com/juandavidvargas19/MAPS_PROJECT/blob/main/SARL/SARL_Training_Standard.sh>. |

## Why this matters

`SARL_Training_Standard.sh` is the shell entrypoint for the paper's SARL runs.
It invokes **`maps_v1.py`** (not `maps_v2.py`), with `base=2000000` frames and
`-ema 25` (= α=0.25). Therefore `maps_v1.py` is the canonical SARL source for
reproduction purposes, regardless of the misleading version suffix.

In `8aa1138` (2026-04-17) the file `maps_v1.py` was deleted from this repo
during a consolidation step justified by `docs/TODO.md` entry **TD-008**, which
incorrectly claimed `maps_v2.py` was "near-identical" to `maps_v1.py`. The
two files in fact diverge structurally — see
`docs/reproduction/sarl-postmortem-20260524.md` for the full diff and the
chain of consequences. This directory restores the lost reference so that
parity tests and future SARL work can be re-anchored on the correct source.

## How to consume this

- **Do not edit these files.** They are read-only reference material.
- The Sprint-04b parity tests under `tests/parity/sarl/` currently target
  `maps_v2.py`. They will need to be re-pointed at `maps_v1.py` before any
  further SARL reproduction work.
- The hyperparameter deviations log under `docs/reproduction/deviations.md`
  was written against `maps_v2.py` as the reference. Several entries may
  need to be re-evaluated against `maps_v1.py` — see the post-mortem doc.

## Cross-references

- Post-mortem: `docs/reproduction/sarl-postmortem-20260524.md`
- Original consolidation commit: `8aa1138`
- TD-008 (the audit that justified the bad consolidation):
  `docs/TODO.md`, entry "TD-008"
- Sprint-04b parity infra: `tests/parity/sarl/`,
  `tests/parity/sarl/_reference_sarl.py`
