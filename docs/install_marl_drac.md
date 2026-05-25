# MARL install on DRAC Tamia (Phase E.6)

**Status :** ✅ Verified 2026-04-21 on Tamia login node.

## Problem

`dm-meltingpot` depends on `dmlab2d`, which **does not publish a Python 3.12
wheel** (only `cp38`, `cp39`, `cp310`, `cp311`). The main MAPS venv uses
Python 3.12, so `uv pip install dm-meltingpot` fails there.

## Solution : dedicated Python 3.11 venv for MARL

Keep the main `.venv` (Python 3.12) for SARL / SARL+CL / Blindsight / AGL.
Create a separate `.venv-marl` (Python 3.11) just for MARL.

### One-time setup (on Tamia login node)

```bash
cd $HOME/links/projects/aip-gdumas85/rram17/Workspace/MAPS/MAPS_PROJECT

# 1. Create the 3.11 venv at $SCRATCH (per lab storage discipline)
uv venv --python 3.11 /scratch/r/rram17/maps/venv-marl
ln -sf /scratch/r/rram17/maps/venv-marl .venv-marl

# 2. Install dm-meltingpot (pulls dmlab2d + tensorflow + ray stack + ~115 pkgs)
VIRTUAL_ENV=/scratch/r/rram17/maps/venv-marl \
    uv pip install --python .venv-marl/bin/python dm-meltingpot
```

Wall time : ~5-10 minutes (large package set).

### Per-job SLURM usage

```bash
#!/bin/bash
#SBATCH --account=aip-gdumas85
#SBATCH --time=...
#SBATCH --mem=...
#SBATCH --gres=gpu:1    # for training

set -euo pipefail

module load StdEnv/2023 python/3.11.5

export VIRTUAL_ENV=/scratch/r/rram17/maps/venv-marl
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Optional: silence TF oneDNN warnings
export TF_ENABLE_ONEDNN_OPTS=0

python scripts/run_marl.py ...
```

### Smoke test

```bash
sbatch --account=aip-gdumas85 --time=00:05:00 --mem=4096M --cpus-per-task=2 \
       --job-name=mp_smoke \
       --output=logs/slurm/mp_smoke_%j.out \
       --error=logs/slurm/mp_smoke_%j.err \
       scripts/slurm/marl_smoke.sh
```

Expected output :
```
Python 3.11.4
--- import dmlab2d ---
/scratch/r/rram17/maps/venv-marl/lib/python3.11/site-packages/dmlab2d/__init__.py
--- import meltingpot ---
/scratch/r/rram17/maps/venv-marl/lib/python3.11/site-packages/meltingpot/__init__.py
--- substrate build ---
roles: ('default', 'default', 'default')
reset OK, num agents: 7
=== E.6 SUCCESS ===
```

## Installed versions (2026-04-21)

| Package | Version | Source |
|:--|:--|:--|
| Python | 3.11.4 | `/cvmfs/soft.computecanada.ca` (loaded via `module load python/3.11.5`) |
| dm-meltingpot | 2.4.0 | PyPI |
| dmlab2d | 1.0.0 (cp311 wheel) | PyPI |
| tensorflow | 2.21.0 | PyPI (transitive via meltingpot) |
| ray | (not pulled — standalone marl install) | — |
| dm-env | 1.6 | PyPI |
| chex | 0.1.91 | PyPI |
| Total pkgs | 115 | — |

**Note :** `ray[default,tune]` is in the main `pyproject.toml` `marl` extras but
is not required for the MeltingPot substrate itself. It will be added to
`.venv-marl` if our port needs it (to be determined at E.8+).

## Troubleshooting

| Symptom | Cause / fix |
|:--|:--|
| `ModuleNotFoundError: No module named 'dmlab2d'` | Using wrong venv — make sure `VIRTUAL_ENV=/scratch/r/rram17/maps/venv-marl` and `PATH` points to its `bin/` first. |
| `module: not found` in sbatch `--wrap` | sbatch `--wrap` uses `/bin/sh` not bash. Use a dedicated `.sh` file with `#!/bin/bash` shebang, not `--wrap`. |
| `Could not find cuda drivers on your machine, GPU will not be used` | Benign on CPU-only sbatch (smoke test doesn't request GPU). Add `--gres=gpu:h100:1` for training. |
| `WARNING: All log messages before absl::InitializeLog() is called are written to STDERR` | Benign TF/abseil initialization noise. |
| Substrate build reports N agents ≠ paper §A.4 | `default_player_roles` may differ from paper setup. Inspect `cfg.default_player_roles` and pass explicit `roles` list to match paper num_agents (to be configured per substrate in E.10). |

## Known paper §A.4 vs substrate mismatches (to verify during E.10)

| Substrate | Paper §A.4 `num_agents` | MeltingPot default |
|:--|:--:|:--:|
| `commons_harvest__closed` | 6 | **7** (from smoke test) — **mismatch** |
| `commons_harvest__partnership` | 4 | TBD |
| `chemistry__three_metabolic_cycles_with_plentiful_distractors` | 8 | TBD |
| `territory_inside_out` | 5 | TBD |

We'll pass explicit `roles=['default'] * num_agents` to `substrate.build()` per
paper §A.4 to force the correct count. To be confirmed at E.10.

## Why a separate venv vs upgrading the main Python

- Main venv is Python 3.12 ; our 258/258 tests for SARL/SARL+CL/Blindsight/AGL
  all pass on that stack.
- Downgrading to 3.11 would require re-syncing + re-running the entire test
  suite + risk accidental divergence on numerics.
- The two-venv approach is SAFE : MARL sbatch jobs activate `.venv-marl`,
  everything else activates `.venv`. No interaction.
- Trade-off : `.venv-marl` duplicates the Python stack (numpy, pytorch, etc).
  Storage cost ~2-3 GB on `$SCRATCH`. Acceptable.

## References

- DRAC Tamia cluster (aip-gdumas85 allocation, Compute Canada)
- MeltingPot 2.x : https://github.com/google-deepmind/meltingpot
- dmlab2d : https://github.com/google-deepmind/lab2d
