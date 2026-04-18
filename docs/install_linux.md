# Install on Compute Canada Narval (Linux + CUDA)

Narval-specific setup for the SARL (Sprint 07) and MARL (Sprint 08)
reproduction sprints. Mac-local installation is covered by the root
`README.md`.

## One-time setup

```bash
# 1. SSH to Narval with MFA.
ssh <username>@narval.computecanada.ca

# 2. Clone (or update) the repo in $HOME.
cd $HOME && git clone https://github.com/Ramdam17/MAPS_PROJECT.git
cd MAPS_PROJECT

# 3. Load Python 3.12. `module spider` to verify the version is available.
module load python/3.12

# 4. uv (lab standard). If not available globally, install to ~/.local/bin.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 5. Sync dependencies — SARL + dev. MinAtar is vendored under external/MinAtar
#    so nothing to download from outside the repo.
uv sync --extra sarl --extra dev
```

## Per-submission workflow

```bash
# 0. cd to the repo (SLURM jobs use ${SLURM_SUBMIT_DIR} as cwd).
cd $HOME/MAPS_PROJECT

# 1. Make sure your branch is up to date and working tree is clean.
git status && git pull --rebase

# 2. Submit the SARL array (150 cells).
sbatch --account=<your-rrg-group> scripts/slurm/sarl_array.sh

# 3. Note the returned ARRAY_JOB_ID, then queue aggregation with a dep:
sbatch --account=<your-rrg-group> \
       --dependency=afterok:<ARRAY_JOB_ID> \
       scripts/slurm/aggregate.sh

# 4. Monitor.
squeue --me
tail -f logs/slurm/maps-sarl-<JOB_ID>_*.out
```

## Output layout

- Per-cell artifacts: `$SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<SEED>/metrics.json`
- Aggregate step rsyncs those into `outputs/sarl/` in the repo working tree
  and runs `scripts/aggregate_sarl.py` → `outputs/reports/sarl_summary.{json,md}`.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `ModuleNotFoundError: minatar` | Run `uv sync --extra sarl` — MinAtar is an optional extra. |
| `CUDA out of memory` | Reduce `training.batch_size` via a `-o` override in the sbatch script. Paper default is 128. |
| `device: cuda` but job runs slow | Check `nvidia-smi` in the sbatch output — if the GPU is idle the tensors may not have moved. Verify `device=cuda` override actually lands in the config. |
| SLURM task indices don't cover 150 cells | Check `--array=0-149` in `sarl_array.sh` matches the game×setting×seed count. |

## Version / hardware assumed

- Narval cluster (AMD EPYC + A100)
- Python 3.12 (`module load python/3.12`)
- PyTorch with CUDA 12.x (installed via `uv sync`)
- MinAtar vendored at `external/MinAtar` (no network fetch needed)
