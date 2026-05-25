# SARL — `maps_v1.py` vs `maps_v2.py`

**Date:** 2026-05-24
**Context:** Natalie Kastel's May 2026 audit of the SARL non-reproduction.
**Status:** Open — port currently anchored on the wrong variant.

## Finding

Juan's fork (`main` branch) ships **both** `maps_v1.py` and `maps_v2.py` under
`SARL/MinAtar/examples/`. The shell entrypoint `SARL/SARL_Training_Standard.sh`
invokes **`maps_v1.py`** with `base=2000000` frames and `-ema 25` (α=0.25).
That makes `maps_v1.py` the canonical SARL source for the paper's Table 6
numbers, regardless of the version suffix.

Our port (`src/maps/experiments/sarl/`) and the Sprint-04b parity reference
(`tests/parity/sarl/_reference_sarl.py`) target **`maps_v2.py`**. They were
ported from `maps_v2.py` after commit `8aa1138` (2026-04-17) renamed it to
`maps.py` and deleted `maps_v1.py`, justified by `docs/TODO.md` entry
**TD-008** ("near-identical, v2 adds only `count_parameters` and
`curriculum`"). That audit was a token-presence comparison, not a structural
diff, and it was wrong.

## Architectural diff (the load-bearing parts)

### QNetwork

| Aspect | `maps_v1.py` (canonical) | `maps_v2.py` (ported) |
|---|---|---|
| Reconstruction decoder | `self.fc_output = nn.Linear(128, 1024)` — dedicated layer, ~131K params | Tied weights: `f.linear(Hidden, self.fc_hidden.weight.t())`, 0 new params |
| Q-head input dim | `nn.Linear(1024, num_actions)` — Q from 1024-dim reconstructed Output | `nn.Linear(128, num_actions)` — Q from 128-dim Hidden |
| First-order cascade target | 1024-dim Output | 128-dim Hidden |
| Return tuple 4th slot | Output (1024-dim) | Hidden (128-dim) |

### SecondOrderNetwork

| Aspect | `maps_v1.py` (canonical) | `maps_v2.py` (ported) |
|---|---|---|
| `comparison_layer` | `nn.Linear(1024, 1024)` active, ~1.05M params | `nn.Linear(1024, 128)` commented out |
| Forward pre-cascade | `dropout(relu(comparison_layer(comparison_matrix)))` | `dropout(comparison_matrix)` direct |

### Training shell

| Param | `SARL_Training_Standard.sh` | port (post-D.12) |
|---|---|---|
| `num_frames` | 2,000,000 (`base=2000000`) | 500,000 (paper Table 11) |
| EMA α | 0.25 (`-ema 25`) | 0.45 (paper Table 11; pre-D.2 was 1.0) |
| Cascade iters | 50 (`-cascade 50`) | 50 (matches) |

## Restored references

The deleted files are now back at `external/paper_reference/sarl/`
(read-only), recovered from `git show 8aa1138^:...`:

- `external/paper_reference/sarl/maps_v1.py` (2,673 lines)
- `external/paper_reference/sarl/SARL_Training_Standard.sh` (77 lines)
- `external/paper_reference/sarl/PROVENANCE.md` (provenance + GitHub URLs)

These are also live at:
- <https://github.com/juandavidvargas19/MAPS_PROJECT/blob/main/SARL/MinAtar/examples/maps_v1.py>
- <https://github.com/juandavidvargas19/MAPS_PROJECT/blob/main/SARL/SARL_Training_Standard.sh>

## Impact on prior conclusions

- **Sprint-04b parity verdict** (port is bit-faithful at atol=1e-6): still
  true, but against `maps_v2.py`. Needs to be re-asserted against
  `maps_v1.py`.
- **Sprint-08 D.1–D.22** (paper-vs-code audit): comparison was paper vs
  `maps_v2.py`. The relevant comparison for reproduction is paper vs
  `maps_v1.py`. Several entries (e.g. `D-sarl-num-frames`, `D-sarl-alpha-ema`)
  may invert their conclusions if v1 is treated as the operative reference.
- **Phase F runs** (90 cells × 500k frames, near-zero returns): mechanism is
  now clear. The port differs from v1 on architecture (QNetwork output dim,
  comparison layer presence, cascade target), and 500k is below the
  convergence horizon Juan used (2M). Re-running on Tamia at 2M frames
  against the current v2-based port will not reproduce the paper —
  re-porting to v1 first is the bottleneck.
- **`juan_sarl_results.csv`** (Natalie's extraction from Juan's
  `processed_data`): reproduces paper Table 6 at high fidelity because it
  comes from `maps_v1.py` runs at 2M frames. This is the only path to
  TMLR/NMI-deadline SARL numbers.

## What to do next (decisions for Rémy + Guillaume)

1. Add `D-sarl-wrong-variant` to `docs/reproduction/deviations.md`
   (severity CRITICAL, root cause = TD-008 sloppy audit) — not auto-applied
   in this commit since `deviations.md` has unrelated pending changes.
2. Mark TD-008 retracted in `docs/TODO.md` with a pointer to this note.
3. For the journal submission: use `juan_sarl_results.csv` as ground truth
   and rewrite the methods section against `maps_v1.py` (2M frames,
   α=0.25, dedicated decoder, `comparison_layer` active, Q from 1024-dim).
4. Re-port to v1 — out of scope for the NMI deadline, schedule a
   post-submission sprint.

## How TD-008 went wrong (one-liner for the next audit)

A token-presence diff on two 2,700-line files is not a structural diff. Any
keep/delete decision between similar files needs `diff -u` plus a check of
which file the live entrypoints invoke. Both checks would have caught this
in under five minutes.
