# Sprint 08 — Reproduction: Perceptual closeout (Blindsight + AGL)

**Status:** ✅ completed 2026-04-21
**Branch:** `repro/paper-faithful`
**Owner:** Rémy Ramadour
**Actual effort:** ~4 days engineering + ~3h DRAC compute wall (400k+ total seed-runs)
**Depends on:** Sprint 06 ✅ (perceptual eval path) · Sprint 07 ✅ (SARL reproduction) · `refactor/clean-rerun` foundation

---

## Context

Sprint 06 shipped the Blindsight + AGL evaluation paths, but left two
reproduction gaps open :

- **RG-002 (Blindsight)** — port main-task acc 0.76 (z=+0.40) vs paper 0.97 (z=9.01).
  Wager 0.71 vs paper 0.85.
- **RG-003 (AGL)** — downstream `training()` phase missing entirely ; port
  only had `pre_train` + single-network eval. No paper-comparable numbers.

Sprint 08 was the end-to-end diagnosis and fix for both gaps, validated on
500-seed arrays per setting and accompanied by full ablation sweeps.

## DoD (checked)

- [x] Paper Table 5a Blindsight reproduced within paper std on both discrim +
  wager metrics, all settings.
- [x] Paper Table 5b/5c AGL reproduced within paper std on both tiers
  (High / Low Awareness), all port-covered settings.
- [x] All architectural deviations (D-blindsight-*, D-agl-*, D-001, D-004)
  either resolved or explicitly documented with ablation evidence.
- [x] Parity tests (bit-exact pretrain, bit-exact training, end-to-end smoke)
  on both Blindsight and AGL.
- [x] Full test suite green (258/258).
- [x] Resolution docs for both RG-002 and RG-003.

## Headline results

### Blindsight (RG-002 resolved in D.25)

| Metric (suprathresh.) | Pre-D.25 port | Post-D.25 port | Paper T.5a |
|:--|:--:|:--:|:--:|
| Discrimination acc. | 0.755 | **0.94 ± 0.03** | 0.97 ± 0.02 |
| Wager accuracy | 0.71 | **0.82 ± 0.04** | 0.85 ± 0.04 |

**Root causes :** two paper↔code discrepancies.
1. Paper T.9 says `hidden_dim=60`, student `main()` literally uses 40.
2. Paper §2.2 cites Pasquali & Cleeremans 2010 for the 2nd-order (which has a
   hidden layer), but student code receives `hidden_2nd=100` as arg and drops
   it — no hidden layer instantiated. Restored in port.

See `docs/reviews/rg002-wager-gap-investigation.md`.

### AGL (RG-003 resolved in D.28)

MAE over 12 metrics (4 settings × 2 tiers × {precision, wager}) :

| Config | MAE vs paper |
|:--|:--:|
| Pre-D.28 (pretrain only, no downstream training) | N/A (no paper-comparable metrics) |
| D.28 Phase B (paper T.10 literal) | 0.0206 |
| **D.28 final (A3 adopted : n_epochs_pretrain=30)** | **0.0142** |

**Root causes :**
1. Missing 3-phase protocol (pretrain → Grammar-A training on 20-cell pool → test).
2. Same Pasquali 2010 hidden layer dropped as Blindsight.
3. Paper T.10 says `n_epochs_pretrain=60`, student `init_global` uses 30 ;
   ablation A3 showed student value gives lower MAE.
4. RangerVA optimizer (paper T.10) confirmed essential via ablation A4.

See `docs/reviews/rg003-resolution.md` and `docs/reviews/rg003-ablation-sweep.md`.

## Sub-phases executed

| Phase | Scope | Commit(s) |
|:--|:--|:--|
| D.1-D.22 | Audits of SARL / SARL+CL / losses / config / deviations catalogue | (pre-Sprint-08 foundation) |
| **D.23** | Review `blindsight/trainer.py` (RG-002 diagnostic) | 2abb936 |
| **D.24** | Review `blindsight/data.py` (bit-parity confirmed) | 2bf9e70 |
| **D.25** | RG-002 fix — `hidden_dim=40` + Pasquali `second_order.hidden_dim=100` | 6211c7f |
| **D.26** | Review `agl/trainer.py` (RG-003 structural diagnosis) | 6713ee2 |
| **D.27** | Review `agl/data.py` (bit-parity confirmed) | 23b423f |
| **D.28.a** | AGL config align paper T.10 + `torch-optimizer` dep | 94fee95 |
| **D.28.b** | Port `AGLTrainer.training()` Grammar-A fine-tuning | b7cc42c |
| **D.28.c** | `AGLNetworkPool` + `NetworkCell` 20-copy replication | fad056f |
| **D.28.d** | `evaluate_pool()` + tier aggregation (High/Low/overall) | b8d999e |
| **D.28.e** | `run_agl.py` full 3-phase pipeline + `--output-dir` | f9b37ac |
| **D.28.f** | Parity tests (bit-exact training, atol=1e-5) | 38b096c |
| **D.28.g+h** | Phase B paper-faithful 500-seed + Phase C 5-ablation sweep (12k runs) | a75e4db |
| **D.28.i** | Adopt A3 defaults + RG-003 resolution doc | a75e4db |
| **D.29** | End-to-end integration test + full suite fix (258/258 green) | ece057a |
| **D.30** | Phase D closeout (this sprint doc + TODO.md update) | (this commit) |

## Key scientific findings

1. **Student code > paper tables** for reproduction (consistent across domains).
   Both Blindsight (T.9 hidden_dim) and AGL (T.10 n_epochs_pre) show that the
   paper's summary tables differ from the code that produced the paper numbers.
   When they disagree, trust the code.

2. **Pasquali 2010 hidden layer was dropped in published code.** Both Blindsight
   and AGL `SecondOrderNetwork.__init__` receive `hidden_second`/`hidden_2nd`
   as argument but never instantiate the layer. Paper §2.2 explicitly cites
   Pasquali & Cleeremans 2010 which has the hidden layer. Restoring it
   materially improves reproduction (closes wager gap in both domains).

3. **`meta = False` override in student AGL `training()` L969 is intentional.**
   Initially hypothesized as a bug (parameter inconsistent with body).
   Ablation A1 proved flipping it **degrades** paper match by 19% MAE. Student
   behavior (2nd-order frozen during Grammar-A fine-tuning) is the correct
   paper-reproduction choice.

4. **High/Low Awareness split is per-seed, not seed-pool.** Initial port
   docstring (`evaluate()` L346) wrongly described the split as a post-hoc
   seed-pool operation. Paper protocol is : each seed replicates its network
   20×, first 10 train for 12 epochs (High), last 10 for 3 (Low), then all 20
   evaluated and averaged per tier. Fixed in D.28.d.

5. **RangerVA is non-negotiable for AGL reproduction.** ADAMAX fallback gives
   MAE 0.0381 ; RangerVA gives 0.0142. `torch-optimizer` is now a hard dep in
   `agl` extras.

## Deliverables

### Code
- `src/maps/experiments/agl/trainer.py` — `training()`, `evaluate_pool()`,
  helpers `_run_training_loop`, `_evaluate_single_cell`, `_aggregate_pool_metrics`.
- `src/maps/experiments/agl/pool.py` (new) — `NetworkCell`, `AGLNetworkPool`.
- `src/maps/experiments/blindsight/trainer.py` — `n_wager_units=2` branch,
  F.BCE-with-logits path.
- `src/maps/components/second_order.py` — `WageringHead(hidden_dim=...)`,
  optional hidden MLP layer.
- `scripts/run_agl.py` — 3-phase pipeline + `--output-dir`.
- `scripts/run_blindsight.py` — `--output-dir`.
- `config/training/{blindsight,agl}.yaml` — paper-faithful defaults with
  detailed rationale comments for every deviation.
- `pyproject.toml` — `torch-optimizer` in `agl` extras.

### Tests
- `tests/parity/_reference_agl_training.py` (new) — student L904 mirror.
- `tests/parity/test_agl_training.py` (new) — bit-exact 4 settings.
- `tests/unit/experiments/agl/test_pool.py` (new) — 8 tests.
- `tests/integration/test_agl_full_pipeline.py` (new) — 6 end-to-end tests.

### Docs
- `docs/reviews/blindsight-eval-metric-audit.md` (D.25 H4 audit).
- `docs/reviews/rg002-wager-gap-investigation.md` (D.25 full RG-002 story).
- `docs/reviews/agl-trainer.md` (D.26 audit).
- `docs/reviews/agl-data.md` (D.27 audit).
- `docs/reviews/rg003-resolution.md` (D.28 RG-003 overview).
- `docs/reviews/rg003-ablation-sweep.md` (D.28.h full sweep detail).
- `docs/plans/plan-20260420-d28-rg003-fix.md` (D.28 execution plan v2).
- `docs/reproduction/deviations.md` §B.9 + §B.10 — all RG-002 + RG-003 rows
  marked ✅ resolved with links to final config + ablation evidence.

### DRAC artefacts (preserved for reproducibility)
- `/scratch/r/rram17/maps/outputs/blindsight/{neither,both}/` — Blindsight ref + fixes.
- `/scratch/r/rram17/maps_exp[A-F]_*/blindsight/` — Blindsight 6-knob sweep.
- `/scratch/r/rram17/maps_agl_d28_paperfaithful/` — AGL Phase B (2000 runs).
- `/scratch/r/rram17/maps_agl_d28_A[1-5]_*/` — AGL Phase C ablations (5 × 2000 runs).

**Total runs on DRAC across Sprint 08 : ~25k.**

## Remaining follow-ups (out of scope, Sprint 08+)

- **D-001 (2-unit wager)** partially closed : implementation supports
  `n_wager_units=2` via port, but AGL's loss/eval branching not plumbed.
  Deferrable — Blindsight D.25 showed 2-unit ≡ 1-unit numerically for 1-hot
  binary targets.
- **D-002 (SimCLR first-order loss)** : still stubbed. Not a gap cause for
  either Blindsight or AGL.
- **Port scope vs paper 6 settings** : our `cascade` boolean applies to both
  networks together. Paper has settings 4 (cascade 1st only + 2nd) and 5
  (cascade 2nd only + 2nd) which port doesn't cover. Known limitation.
  Not critical for reproduction (settings 1/2/3/6 already match paper).
- **Cross-domain RangerVA decision for Blindsight** : currently only AGL uses
  RangerVA. Blindsight uses ADAMAX (student actual + paper T.9). If we
  generalize, could re-evaluate.

## Retrospective

### What went well
- **Micro-sprint discipline** (9 D.28 sub-phases, each with Orient→Do→Verify→
  Report→Commit→Wait) kept the scope under control despite the 500-seed ×
  multi-ablation complexity.
- **Paper-first audit** (D.26 → D.28 plan) caught the 3 paper↔code
  discrepancies before launching compute. Saved ≥ 1 day of wasted runs.
- **Bit-exact parity tests** on both pretrain and training phases gave
  confidence at each refactor step. Without them, `_run_training_loop`
  extraction would have been scary.
- **Ablation sweep framework** (5 knobs × 4 settings = 20 arrays chained via
  `--dependency`) validated every design decision empirically rather than by
  assertion. The `meta=False` hypothesis was disproved, and A3 `n_pre=30`
  was adopted only after measuring MAE improvement.

### What to retain
- **Always chain sbatch with `--dependency=afterany` when > 3 jobs.** Avoids
  monopolizing the shared lab queue. Memorized this sprint.
- **`--mem=2048M` not `--mem=2G`** to silence DRAC SLURM warning. Memorized.
- **When paper table and code disagree, benchmark both.** Paper T.9 vs student
  `main()` hidden_dim was resolved by ablation. Same for paper T.10 vs
  `init_global` n_epochs. This is a pattern, not a one-off.
- **Extract shared training loop into module-level helper (`_run_training_loop`)**
  when two callers (trainer.training + pool.train_range) need the same body.
  Avoids duplication and keeps parity test surface small.

### What to improve
- **Scope creep in D.28 planning** : initial v1 was 7 sub-phases, grew to 9
  after Rémy's feedback. Should have anticipated ablation complexity upfront.
- **Compute estimate wrong by 10×** : initial plan said 20-60h DRAC wall,
  actual was ~3h. Under-estimated parallelism. Fixed in plan v2.
- **Submission mistake early in D.28.h** : accidentally submitted 20 arrays
  at once without dependency, had to cancel and re-chain. Respect the
  shared queue discipline more rigorously.

---

**Sprint 08 closes with Phase D fully complete. Both perceptual reproduction
gaps (RG-002 Blindsight, RG-003 AGL) resolved with paper-faithful port
configurations and 500-seed validation on DRAC. Ready for next sprint work
(potentially Phase E : MARL / MeltingPot reproduction, or Phase F : full
experiment matrix regeneration).**
