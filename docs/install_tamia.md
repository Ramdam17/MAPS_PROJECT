# Install on Tamia (DRAC) — MAPS

**Cluster:** Tamia (DRAC) · **User:** rram17 · **Allocation:** `aip-gdumas85` · **Last verified:** 2026-04-18 (Sprint 07 Phase 0-1).

Tamia differs from Narval in ways that surprised the Sprint 07 plan draft — the sections below capture every delta so future sessions can start with the correct assumptions.

---

## Quick-start (reference)

```bash
# 1. login node (tamiaN.tamia.ecpia.ca)
cd /project/6102289/rram17/Workspace/MAPS/MAPS_PROJECT
module load StdEnv/2023 python/3.12 cuda/12.6
export PATH="$HOME/.local/bin:$PATH"          # uv

# 2. (one-time) install uv + sync venv — ONLY from login, compute has no net
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra sarl --extra dev              # torch cu128 wheel → ~6 min

# 3. submit — never run pytest / training directly on login
sbatch scripts/slurm/smoke_sarl.sh breakout 6 42 50000 cpu
```

---

## Allocation

- **Account:** `aip-gdumas85` — **single account**, no `_cpu`/`_gpu` suffix like Narval. Any sbatch with `--account=aip-gdumas85_cpu` or `_gpu` will fail (`Invalid account`).
- **QOS:** `normal`.
- `sshare -U -u rram17` (2026-04-18):

  ```
  Account         User     RawShares  NormShares  FairShare
  aip-gdumas85    rram17           1    0.066667   0.350379
  ```

---

## Partitions (Tamia, 2026-04-18)

| Partition | GRES | MaxTime | Notes |
|---|---|---|---|
| `cpubase_interac` | — | 6 h | `QoS=interac` |
| `cpubase_bynode_b{1,2,3}` | — | 3 h / 12 h / **24 h** | Full-node CPU allocation |
| `cpubase_bycore_b{1,2,3}` | — | 3 h / 12 h / **24 h** | Per-core (use for small CPU jobs) |
| `gpubase_interac` | `gpu:h100:4` or `gpu:h200:8` | 6 h | Interactive |
| `gpubase_bynode_b{1,2,3}` | `gpu:h100:4` or `gpu:h200:8` | 3 h / 12 h / **24 h** | Whole-node GPU |

**Don't specify `--partition` in sbatch.** The scheduler routes automatically based on resources and `--time`; forcing a partition often returns `The specified partition does not exist, or the submitted job cannot fit in it`. Submit without `--partition` and it just works.

### GPU: whole-node allocation, no MIG

- `GresTypes=gpu` only — **MIG is not configured.** Nodes expose whole GPUs as `gpu:h100:4` (SXM, 4 GPUs/node) or `gpu:h200:8` (8 GPUs/node). You cannot request a single H100.
- **Request with `--gpus-per-node=h100:4`** (or `h200:8`). `--gres=gpu:h100:1` fails with `The h100 GPUs are only allocated by node`.
- Driver/runtime: NVIDIA 570.211.01 / CUDA 12.8. Install torch wheels compiled for `cu128` (see pyproject pin) — the default `+cu130` wheels from PyPI fail with `driver too old`.
- Capability 9.0 (Hopper SXM). 80 GB HBM3 / GPU.
- For SARL (MinAtar, small model): one whole H100 node is heavy overkill; the bench in Phase 2 decides whether GPU is worth the per-cell cost vs CPU.

---

## Filesystem & quotas

`diskusage_report` on 2026-04-18:

```
/home   (user rram17)        11 GiB / 25 GiB        52K / 250K files
/scratch (user rram17)        4 B   / 1 TiB          1 / 1M files
/project (aip-gdumas85)      1.1 TiB / 2 TiB        46K / 500K files
```

- `$HOME` = `/home/r/rram17` — uv / `~/.local/bin/uv`, shell rc, modest size.
- `$SCRATCH` = `/scratch/r/rram17` — **write heavy outputs here** (300-cell SARL array: ~3 GB total). Purged on an auto-cleanup policy; copy results back to `/project` in Phase 5.
- `$PROJECT`-ish = `/project/6102289/rram17/Workspace/MAPS/MAPS_PROJECT` — code + configs + committed docs. Slower (NFS); do not write training outputs here.
- Our `config/paths.yaml` + `maps.utils.paths.get_paths()` read `$SCRATCH` automatically into `paths.scratch_root`; sbatch scripts write to `${SCRATCH}/maps/outputs/...`.

---

## Modules

```bash
module load StdEnv/2023 python/3.12 cuda/12.6
```

- `StdEnv/2023` is sticky (usually pre-loaded). `python/3.12.4` and `cuda/12.6` are the current defaults — `cuda/12.2` also available.

---

## uv + venv

**Rule: `uv sync` runs on the login node only.** Compute nodes have no outbound internet — a bare `uv sync` on a compute node times out downloading wheels. The venv under `.venv/` lives on the shared FS and is visible to compute jobs offline.

```bash
# one-time
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# sync deps — from login node
uv sync --extra sarl --extra dev

# compute nodes: always `--offline` to fail loud if something tries to fetch
uv run --offline pytest ...
uv run --offline python scripts/run_sarl.py ...
```

### torch cu128 pin

`pyproject.toml` declares a cu128 index for torch on Linux (see `[tool.uv.sources]`). This matches the Tamia driver CUDA 12.8 runtime exactly. macOS/Windows fall back to PyPI.

---

## sbatch template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my-job
#SBATCH --account=aip-gdumas85          # no _cpu/_gpu suffix
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M                     # M > G avoids the G=1024M warning
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# source scripts/slurm/common.sh for modules + venv + traps
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

uv run --offline python your_script.py
```

For GPU work, add `--gpus-per-node=h100:4` at submit time (or `h200:8`). Don't put `--partition` — scheduler picks the right tier from `--time`.

### Signal handling

Tamia pre-emption is rare on `aip-*` but possible; all production SARL sbatch scripts use `--requeue` + `common.sh` traps to log wall-clock on SIGTERM/SIGUSR1 for post-mortem.

---

## Login node hygiene

**Never run `pytest`, `uv sync`, `torch` imports, or any script with measurable CPU on a login node.** The login nodes are shared by ~13 users; one pytest run at 48 % of a core takes the load average to 30 and makes everyone's shell laggy. Use `sbatch` or `salloc` for anything compute-ish, even a 10-second smoke test.

For quick interactive debugging:

```bash
salloc --account=aip-gdumas85 --time=01:00:00 --cpus-per-task=2 --mem=8000M
```

---

## Tamia-vs-Narval cheatsheet

| | Narval | Tamia |
|---|---|---|
| Account split | `rrg-xxx_{cpu,gpu}` / `def-xxx_{cpu,gpu}` | **single** `aip-gdumas85` |
| MIG | some H100 partitions expose slices | **not available** |
| GPU request | `--gres=gpu:1` works | must use `--gpus-per-node=h100:4` (or `h200:8`) |
| Partitions | explicit usually required | **omit** `--partition`, scheduler routes |
| Driver | varies | 570.x / CUDA 12.8 |
| torch wheel | cu121/cu124 commonly | **cu128** required |

---

## Smoke-test checklist (Sprint 07 Phase 0)

| Check | How | Expected |
|---|---|---|
| SSH | `ssh -T git@github.com` | "Hi Ramdam17!" |
| Allocation | `sshare -U -u rram17` | `aip-gdumas85`, FairShare > 0 |
| Modules | `module load StdEnv/2023 python/3.12 cuda/12.6` | no error |
| uv | `uv --version` | 0.11+ |
| venv + torch | `uv run --offline python -c "import torch; print(torch.__version__)"` (on compute) | `2.11.0+cu128` |
| GPU | `uv run --offline python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` (on GPU node) | `True 4` |
| pytest | `sbatch` pytest unit+parity | 169 passed |
| Training | `sbatch scripts/slurm/smoke_sarl.sh breakout 1 42 10000 cpu` | `metrics.json` produced |
