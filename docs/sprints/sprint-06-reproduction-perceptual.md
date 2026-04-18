# Sprint 06 — Reproduction: Blindsight + AGL (Mac-local)

**Status:** ⚪ planned
**Branch:** `repro/perceptual`
**Owner:** Rémy Ramadour
**Est. effort:** 2-3 days wall-clock (compute-bound, not engineering-bound)
**Depends on:** Sprint 04 ✅ (perceptual splits) · Sprint 05 ✅ (tests + CI) · parity harness green

---

## Context

Sprints 00-05 delivered a refactored, config-first, tested codebase. Sprint 06 is the first **reproduction** sprint: take the paper's perceptual-domain headline numbers (Blindsight detection, AGL high/low awareness) and reproduce them end-to-end on Mac CPU/MPS with the new pipeline.

**This sprint is not an engineering sprint.** No new features. No refactors. The deliverables are: runs + numbers + a report with z-scores.

If the numbers fall outside the paper's ±2σ tolerance, the sprint **fails loudly**: we flag the drift, log it in `docs/reproduction/deviations.md`, and decide whether it's a bug, a seed-control artifact, or a genuine reproduction gap — **we do not silently tune things to make numbers match.**

## Target numbers (from `docs/reproduction/experiment_matrix.md`)

| Task | Metric | Paper mean ± std | Our tolerance (±2σ) |
|------|--------|-----------------:|--------------------:|
| Blindsight | detection acc. | 0.97 ± 0.02 | [0.93, 1.01] |
| AGL — High Awareness | classification acc. | 0.66 ± 0.05 | [0.56, 0.76] |
| AGL — Low Awareness | classification acc. | 0.62 ± 0.07 | [0.48, 0.76] |

**Baseline comparison:** setting "neither" (second-order OFF, cascade OFF). The headline numbers above correspond to the **full MAPS** setting ("both" ON). The z-score comes from the baseline's mean/std — we must produce both.

## Objectives

1. Run Blindsight × 4 settings × 10 seeds (= 40 runs) on Mac CPU.
2. Run AGL × 4 settings × 10 seeds × 2 awareness conditions (= 80 runs) on Mac CPU.
3. Compute per-setting mean/std; compute z-score of full-MAPS vs. baseline; compare against paper's z-score (not just the raw mean).
4. Write `docs/reports/sprint-06-report.md` with a replicable summary (seeds, wall-clock, deviations, pass/fail per cell).

## Non-goals

- No SARL / SARL+CL / MARL runs (those are sprints 07-08, compute-canada).
- No new code beyond small glue (run orchestration, aggregation). **If you find yourself touching `src/maps/experiments/*`, stop and open a tech-debt ticket instead.**
- No hyperparameter tuning. Configs are locked to `config/training/{blindsight,agl}.yaml`.

## Tasks

### 6.1 — Seed + setting matrix

- [ ] Extend `scripts/run_blindsight.py --all-settings` to accept `--seeds "1,2,...,10"` and write each cell to `outputs/blindsight/<setting>/seed-<NN>/`.
- [ ] Same for `scripts/run_agl.py`, with `--awareness {high,low}` already present.
- [ ] Verify `outputs/<domain>/<setting>/seed-<NN>/metrics.json` is written per-cell with the final-epoch accuracy.
- [ ] Dry-run one cell per domain (1 setting × 1 seed) to verify the schema before unleashing the full 120-run grid.

### 6.2 — Run the grid

- [ ] Blindsight: 4 settings × 10 seeds = **40 runs**, estimated ~15 min/run on M-series → ~10 h wall-clock.
- [ ] AGL high-awareness: 4 × 10 = **40 runs**, similar wall-clock.
- [ ] AGL low-awareness: 4 × 10 = **40 runs**.
- [ ] Use tmux for babysitting (see `tmux-pipelines` skill). Single process is fine; MPS does not parallelize cleanly on M-series.
- [ ] Log each run's wall-clock into the per-cell `metrics.json` for later scaling estimates.

### 6.3 — Aggregation

- [ ] `scripts/aggregate_perceptual.py` — walks `outputs/{blindsight,agl}/`, produces:
    - Per-setting table: mean ± std across seeds
    - Z-score of each non-baseline setting vs. baseline's seed distribution
    - Exit code != 0 if any cell is **missing** (no silent partial reports)
- [ ] Writes `outputs/reports/perceptual_summary.json` + a human-readable `.md` table.

### 6.4 — Reporting

- [ ] `docs/reports/sprint-06-report.md` containing:
    - Seed list, git SHA, hardware (M-series model), wall-clock totals
    - Per-cell table: our mean/std, paper mean/std, z-score ours, z-score paper, delta, verdict (✅ within ±2σ / ⚠️ outside)
    - Any deviations logged against `docs/reproduction/deviations.md`
    - Short narrative: did MAPS outperform baseline in our hands, at what effect size?

### 6.5 — Deviation handling (only if needed)

If any cell falls outside ±2σ:
- [ ] Do NOT change hyperparameters to "fix" it.
- [ ] Open a ticket in `docs/TODO.md` under "Reproduction gaps".
- [ ] Log the observation in `docs/reproduction/deviations.md` with: config SHA, seed set, observed vs. expected, hypothesis for the delta.
- [ ] If the delta is large (>4σ), halt the sprint and escalate to Guillaume before continuing.

## Definition of Done

- 120 cells written to `outputs/{blindsight,agl}/` with `metrics.json` per cell
- `outputs/reports/perceptual_summary.json` regenerable from scratch via `scripts/aggregate_perceptual.py`
- `docs/reports/sprint-06-report.md` pushed to main (via feature branch + `--no-ff` merge)
- `docs/reproduction/experiment_matrix.md` updated with "our numbers" column
- All numbers either ✅ within paper's ±2σ, or ⚠️ logged in `deviations.md` with a hypothesis

## Risks

1. **Single-seed noise ≠ paper noise.** Paper's N=10 and our N=10 both have meaningful std. A single cell outside ±2σ is expected at this seed count; two or more is a signal.
2. **MPS nondeterminism.** Run everything on CPU unless a run is too slow to be practical. If MPS is used, note it per-cell.
3. **Config drift between Sprint 04 extraction and paper.** The parity tests (atol=1e-5, Sprint 04) cover the forward/training pass; they don't guarantee the *schedule* (200 epochs × 100 patterns, StepLR 25/0.98) matches the paper's actual runs — we may find a step-count mismatch. If so, log in `deviations.md` and re-check against `BLINDSIGHT/Blindsight_TMLR.py` git history.
4. **Wall-clock overshoot.** 10 h per sub-grid is the optimistic estimate. Budget 2-3 days including restarts, and keep the laptop plugged in.

## Pointers

- Perceptual trainers: `src/maps/experiments/{blindsight,agl}/trainer.py`
- Configs: `config/training/{blindsight,agl}.yaml` (inherit `config/maps.yaml`)
- CLIs: `scripts/run_{blindsight,agl}.py`
- Target matrix: `docs/reproduction/experiment_matrix.md`
- Deviation log: `docs/reproduction/deviations.md`

## Next up

**Sprint 07** — SARL reproduction on Narval (GPU, 5 seeds × 5 envs × 6 settings = 150 runs). Starts once Sprint 06 numbers are pinned and we have a reproducible aggregation pipeline we can port to the cluster.
