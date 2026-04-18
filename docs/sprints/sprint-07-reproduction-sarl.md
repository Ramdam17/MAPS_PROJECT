# Sprint 07 — Reproduction: SARL on Narval

**Status:** 🟡 in progress (scaffolding)
**Branch:** `repro/sarl`
**Owner:** Rémy Ramadour
**Est. effort:** 1-2 weeks compute (scaffolding ~1 day)
**Depends on:** Sprint 04b ✅ (SARL architectural parity) · Sprint 05 ✅ (tests + CI) · Sprint 06 ✅ (perceptual reproduction)

---

## Context

Sprint 04b proved the SARL refactor is *architecturally* faithful (3-tier parity
forward/buffer/update at atol=1e-6). Sprint 07 uses that guarantee to prove
*scientific* parity: the paper's SARL z-scores reproduce within ±2σ on at least
two games.

MinAtar DQN training is compute-bound: 5M frames × 150 runs (5 games × 6 settings
× 5 seeds) exceeds what a Mac CPU can do in wall-clock. Sprint 07 delegates
compute to **Compute Canada Narval** (GPU) via SLURM array jobs.

## Target numbers (from `docs/reproduction/experiment_matrix.md`)

| Env | Metric | Paper mean ± std | Paper z | Our tolerance (±2σ) |
|-----|--------|-----------------:|--------:|--------------------:|
| Seaquest | mean return (final 100 eps) | 3.06 ± 0.34 | 7.03 | [2.38, 3.74] |
| Asterix | mean return | 4.54 ± 1.01 | 1.32 | [2.52, 6.56] |
| Breakout | mean return | 8.07 ± 0.72 | 3.70 | [6.63, 9.51] |
| Space Invaders | mean return | 26.80 ± 1.59 | 4.13 | [23.62, 29.98] |
| Freeway | mean return | 34.20 ± 2.83 | 0.95 | [28.54, 39.86] |

DoD target: **at least 2 environments reproduce within ±2σ**. Seaquest + Breakout
are the highest-z (most differentiated from baseline) and the first priority.

## Objectives

1. Scaffold SLURM orchestration for the full SARL grid: 5 games × 6 settings × 5 seeds = **150 runs** at 5M frames each.
2. Local CPU smoke test (reduced frames) to confirm the pipeline runs end-to-end under the same config path as the cluster job.
3. Aggregation script for SARL outputs (mean return over final 100 eps per cell → per-setting stats → z vs baseline setting 1).
4. Narval runs (compute-heavy) — not blocking for sprint scaffolding.
5. Write `docs/reports/sprint-07-report.md` with z-score verdict per environment.

## Non-goals

- No SARL+CL runs (Sprint 07 is SARL only; CL is a separate sprint if needed).
- No MARL (Sprint 08).
- No changes to `src/maps/experiments/sarl/` — architectural parity is locked.
  If you find yourself editing the trainer, stop and open a ticket.

## Tasks

### 7.1 — Local smoke test (gate before Narval submission)

- [ ] One cell at reduced horizon: `--game breakout --setting 6 --seed 42 --num-frames 50000`
- [ ] Confirms: env loads, replay buffer fills, target net syncs, metrics.json written with `final_100_return` field, checkpoint saved.
- [ ] Wall-clock must be < 30 min on M-series CPU or we need to GPU-accelerate the smoke step.

### 7.2 — SLURM orchestration

- [ ] `scripts/slurm/sarl_array.sh` — sbatch array script for 150 runs (game/setting/seed index → CLI invocation of `run_sarl.py`).
- [ ] `scripts/slurm/aggregate.sh` — runs `scripts/aggregate_sarl.py` after the array finishes.
- [ ] `docs/install_linux.md` — minimal Narval-specific setup (uv / venv / minatar install, dataset-free since MinAtar is self-contained).
- [ ] All SLURM scripts parameterized — no hardcoded `/home/<user>` paths (enforced by pre-commit hook).

### 7.3 — Aggregation

- [ ] `scripts/aggregate_sarl.py`:
    - Walks `outputs/sarl/<game>/setting-<N>/seed-<SEED>/metrics.json`
    - Per (game, setting): mean/std of `final_100_return` across seeds
    - Z-score of each setting vs. setting 1 (vanilla DQN baseline)
    - Exits non-zero on missing cells (same contract as `aggregate_perceptual.py`)
    - Writes `outputs/reports/sarl_summary.{json,md}`

### 7.4 — Narval run (compute, blocking)

- [ ] Submit the array to Narval. Expect ~150 × (5M frames / ~500 FPS on GPU) ≈ 25 GPU-hours per cell × 150 = 3,750 GPU-hours total. Narval's typical allocation accommodates this but not in a single day — budget 5-10 wall-clock days with queue time.
- [ ] Monitor: `squeue`, spot-check 1-2 cells mid-run against local parity expectations.

### 7.5 — Reporting

- [ ] `docs/reports/sprint-07-report.md`:
    - Per-env table: our mean/std/z vs. paper mean/std/z
    - Verdict (✅ within ±2σ / ⚠️ outside) per environment
    - Total GPU-hours used; SLURM job IDs for provenance
- [ ] Update `docs/reproduction/experiment_matrix.md` with our numbers
- [ ] If any cell falls outside ±2σ, log in `docs/reproduction/deviations.md`
      under a new "Sprint 07 reproduction gaps" section. Do NOT tune
      hyperparameters to close the gap.

## Definition of Done

- Scaffolding (7.1-7.3) merged to main, local smoke green
- Narval runs submitted and results collected (7.4) — this is the compute-wait
- `aggregate_sarl.py` regenerable from scratch against `outputs/sarl/`
- `sprint-07-report.md` on main, experiment_matrix.md updated
- **At least 2 of 5 environments reproduce within ±2σ of the paper's reported mean**

## Risks

1. **MinAtar version drift** — pin version in `pyproject.toml` (already done via `external/MinAtar`).
2. **Narval queue times** — the 150-run array may span several days. Partial-result aggregation must be a script, not manual bookkeeping.
3. **Seed-reduction temptation** — paper uses 10 seeds, our scoped DoD uses 5. A single out-of-tolerance cell at N=5 is expected; two or more is a reproduction gap.
4. **GPU vs CPU numerical drift** — parity tests are CPU-only. First Narval cell should be spot-checked against a CPU reference run at the same seed.

## Pointers

- SARL trainer: `src/maps/experiments/sarl/` (parity-green since Sprint 04b §4.5)
- SARL CLI: `scripts/run_sarl.py`
- Config: `config/training/sarl.yaml`
- Paper setting table: 1=vanilla, 2=cascade-FO, 3=meta, 4=meta+cascade-FO, 5=meta+cascade-SO, 6=full MAPS

## Next up

**Sprint 08** — MARL on Narval (MeltingPot 2.0 MAPPO). Can run in parallel with Sprint 07 Narval queue once SLURM infra is in place.
