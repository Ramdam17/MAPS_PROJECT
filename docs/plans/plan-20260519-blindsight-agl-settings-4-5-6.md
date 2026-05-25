# Plan: Blindsight + AGL — port settings 4, 5, 6 to match paper Table 5

**Date:** 2026-05-19
**Sprint:** Sprint 08 (closeout extension)
**Estimated complexity:** L
**Author:** Rémy Ramadour + Claude

---

## Problem Statement

The MAPS paper reports **6 factorial settings** per Know-Thyself domain (Blindsight + AGL,
paper Table 5a/5b/5c). Our current port (`config/experiments/factorial_2x2.yaml`,
`src/maps/experiments/blindsight/trainer.py`, `src/maps/experiments/agl/trainer.py`)
implements a **2×2 = 4-cell factorial** over `(cascade: bool, second_order: bool)`,
applying cascade **symmetrically across both networks** by construction.

Mapping the 4 current cells to paper settings (1-6 per `experiment_matrix.md` §Settings
factorial):

| Current cell        | cascade | 2nd-order | Effective cascade application | = paper Setting |
|---------------------|:-------:|:---------:|:-----------------------------:|:---------------:|
| `neither`           | OFF     | OFF       | none                          | **1** (Baseline) |
| `cascade_only`      | ON      | OFF       | 1st only (2nd is no_grad)     | **2** (Cascade 1st only) |
| `second_order_only` | OFF     | ON        | none                          | **3** (2nd-order no cascade) |
| `both`              | ON      | ON        | **1st AND 2nd**                | **6** (Full MAPS) |

**Critical finding (2026-05-19):** the cell currently *labelled* "Full MAPS"
(`both`) is in fact paper **Setting 6** (cascade applied to both networks), **not
Setting 4** (MAPS, cascade applied to 1st network only). Settings 4 and 5 are
absent from the port; the `knowyourself_outputs_20260501.tar.gz` archive shipped
to Natalie Kastel reflects this 4-cell coverage.

Paper Table 5a/5b headline target is Setting 4 (cascade 1st only, z=9.01 Blindsight,
z=8.20 AGL high, z=15.7 AGL low). Setting 6 numbers happen to be numerically close
in Table 5a (both 0.97 ± 0.02 main task acc), so the current "MAPS" comparison still
passes by coincidence — but the **claim** that we reproduce MAPS Setting 4 is
unsubstantiated.

**Out of scope for this plan** (filed separately): the baseline-shift question
on Blindsight Setting 1 (0.847 vs paper 0.95) reported by N. Kastel — to be
investigated after settings 4, 5, 6 are merged and re-run.

## Scientific Rationale

**Why asymmetric cascade matters.** The paper's central architectural claim is
that *the cascade benefits the 1st-order encoder, while the 2nd-order wager
read-out works better without cascade* (Setting 4 = MAPS). Conflating that with
Setting 6 (cascade-on-both) prevents the port from validating or refuting the
key ablation — and from showing the same z-score pattern across Blindsight, AGL,
and SARL.

**MARL already does this right.** `src/maps/experiments/marl/setting.py:MarlSetting`
exposes `cascade_iterations1` and `cascade_iterations2` as independent integers,
ported from the student `train_meltingpot.py:112-142` `(meta, cascade_iter1,
cascade_iter2)` mapping. The MARL output archive (`marl_outputs_20260428.tar.gz`)
contains all 6 cells with explicit ids `setting-baseline`,
`setting-cascade_1st_no_meta`, `setting-meta_no_cascade`,
`setting-meta_cascade_2nd`, `setting-meta_cascade_both`, `setting-maps`.

**This plan ports the MARL pattern to Know-Thyself**, with the constraint that
existing 4-cell results (and the D.25 RG-002 closeout numbers) must remain
numerically reproducible — the refactor is API-additive, not semantic-changing
for the cells already shipped.

**References.**
- Paper Table 5a/5b/5c — `docs/reproduction/paper_targets_extracted.md` lines 44-73.
- Settings 1-7 canonical labels — `docs/reproduction/experiment_matrix.md` lines 32-50.
- MARL precedent — `src/maps/experiments/marl/setting.py`.
- Existing deviations log — `docs/reproduction/deviations.md` §B.9 (D.25, RG-002).

## Implementation Approach

### Data flow (unchanged at the trainer interface)

```
factorial_6cell.yaml (id, label, cascade_1st: bool, cascade_2nd: bool, second_order: bool)
  ↓ BlindsightSetting.from_dict / AGLSetting.from_dict
  ↓ BlindsightTrainer(cfg, setting) / AGLTrainer(cfg, setting)
  ↓ build() → trainer holds (cascade_iters_1, cascade_rate_1, cascade_iters_2, cascade_rate_2)
  ↓ pre_train() → forward loop runs 1st-order cascade for cascade_iters_1, 2nd-order for cascade_iters_2
Output: $SCRATCH/maps/outputs/{blindsight,agl}/<setting_id>/seed-<seed>/{summary.json, losses_*.npy, *.pt}
```

### Key components

| File | Change | Responsibility |
|------|--------|----------------|
| `src/maps/experiments/blindsight/trainer.py` | refactor `BlindsightSetting` + `BlindsightTrainer.__init__` | hold separate `cascade_iters_1`, `cascade_iters_2`, `cascade_rate_1`, `cascade_rate_2`; forward loop branches |
| `src/maps/experiments/agl/trainer.py` | refactor `AGLSetting` + cascade plumbing in pretrain/train/test phases | same asymmetric cascade as Blindsight, applied to all 3 AGL phases |
| `config/experiments/factorial_2x2.yaml` | **rename** → `factorial_6cell.yaml` + add 6 cells | source of truth for the 6 paper cells |
| `config/experiments/factorial_2x2.yaml` (legacy) | **keep as alias** pointing to `factorial_6cell.yaml` with 4 cells filter | backward compat for any external caller still using the name |
| `scripts/run_blindsight.py` | accept new setting ids + `--all-settings` enumerates 6 cells | CLI parity |
| `scripts/run_agl.py` | same | CLI parity |
| `scripts/aggregate_perceptual.py` | recognise new setting ids in aggregation | so figures pick up settings 4 + 5 |
| `tests/parity/test_blindsight_settings.py` (new) | numerical parity setting 1-3 + 6 against pre-refactor reference | guard the existing archive |
| `tests/parity/test_agl_settings.py` (new) | same for AGL | guard the existing archive |
| `tests/unit/test_setting_dataclasses.py` (new) | validate `from_dict` for both new dataclasses | catch YAML schema drift |
| `docs/reproduction/deviations.md` | new entry `D-blindsight-cascade-symmetry-fixed` + `D-agl-cascade-symmetry-fixed`, plus correct the D.25 closeout narrative ("Setting 6, not Setting 4") | scientific record |
| `docs/reproduction/experiment_matrix.md` | update §Blindsight + §AGL "Current reproduction status" tables to reflect the 6-cell mapping | scientific record |
| `docs/sprints/sprint-08-reproduction-perceptual-closeout.md` | append section "Sprint-08 D.31 — settings 4/5/6 closure" with this plan ref and outcome | sprint trail |

### Config changes needed

```yaml
# config/experiments/factorial_6cell.yaml (new)
# Paper Table 5a/5b/5c — 6 factorial settings for Know-Thyself.
# Maps (cascade_1st, cascade_2nd, second_order) per experiment_matrix.md §Settings factorial.

settings:
  - id: setting-1-baseline
    label: "Baseline (no cascade, no 2nd-order)"
    cascade_1st: false
    cascade_2nd: false
    second_order: false

  - id: setting-2-cascade-1st
    label: "Cascade 1st Net only"
    cascade_1st: true
    cascade_2nd: false
    second_order: false

  - id: setting-3-second-order-only
    label: "2nd-order, no cascade"
    cascade_1st: false
    cascade_2nd: false
    second_order: true

  - id: setting-4-maps
    label: "MAPS (cascade 1st Net + 2nd-order)"
    cascade_1st: true
    cascade_2nd: false
    second_order: true

  - id: setting-5-cascade-2nd
    label: "Cascade 2nd Net only + 2nd-order"
    cascade_1st: false
    cascade_2nd: true
    second_order: true

  - id: setting-6-full-maps
    label: "Full MAPS (cascade both + 2nd-order)"
    cascade_1st: true
    cascade_2nd: true
    second_order: true

n_seeds: 500
seeds: [100, 101, ..., 599]  # canonical 500-seed pool (Sprint-08 D.25, B.9 D-blindsight-seeds)
```

```python
# Dataclass shape (Blindsight + AGL, mirroring MarlSetting)
@dataclass(frozen=True)
class BlindsightSetting:
    id: str
    label: str
    cascade_1st: bool
    cascade_2nd: bool
    second_order: bool

    @classmethod
    def from_dict(cls, d) -> "BlindsightSetting":
        return cls(
            id=str(d["id"]),
            label=str(d.get("label", d["id"])),
            cascade_1st=bool(d["cascade_1st"]),
            cascade_2nd=bool(d["cascade_2nd"]),
            second_order=bool(d["second_order"]),
        )

    # Back-compat helpers for callers that previously read `cascade` / `second_order`.
    @property
    def cascade(self) -> bool:
        """Legacy 2×2 view — True iff either side has cascade on."""
        return self.cascade_1st or self.cascade_2nd
```

Trainer change (Blindsight excerpt; AGL mirrors):

```python
# Asymmetric cascade schedule (paper Table 5, settings 4/5/6).
self.cascade_rate_1 = float(cfg.cascade.alpha) if setting.cascade_1st else 1.0
self.cascade_iters_1 = int(cfg.cascade.n_iterations) if setting.cascade_1st else 1
self.cascade_rate_2 = float(cfg.cascade.alpha) if setting.cascade_2nd else 1.0
self.cascade_iters_2 = int(cfg.cascade.n_iterations) if setting.cascade_2nd else 1

# ... forward loop:
for _ in range(self.cascade_iters_1):
    h1, h2 = self.first_order(batch.patterns, prev_h1=h1, prev_h2=h2, cascade_rate=self.cascade_rate_1)

if self.setting.second_order:
    for _ in range(self.cascade_iters_2):
        wager, comparison = self.second_order(batch.patterns, h2, comparison, self.cascade_rate_2)
    # ... loss_2 backward as before
else:
    with torch.no_grad():
        for _ in range(self.cascade_iters_2):
            _, comparison = self.second_order(batch.patterns, h2, comparison, self.cascade_rate_2)
```

Invariant: when `cascade_1st == cascade_2nd`, behavior is bit-identical to the
pre-refactor `BlindsightTrainer` for the four legacy cells. This is enforced by
the parity tests below.

## Risks & Unknowns

- **Risk 1: numerical drift on legacy cells (1, 2, 3, 6) after refactor.**
  → Mitigation: parity tests load reference seeds from the
  `knowyourself_outputs_20260501.tar.gz` archive (or re-generate from
  pre-refactor commit), run with new code, assert bitwise-equal losses + final
  weights. Block merge if any cell drifts.

- **Risk 2: RNG consumption pattern changes when cascade_2nd=False and
  cascade_1st=True.** The legacy throwaway pass runs the 2nd-order net for
  `cascade_iters` (= 50 when cascade=True, 1 when cascade=False). If we keep
  `cascade_iters_2=1` for new Setting 4 but the legacy `cascade_only` used 50,
  the no_grad pass length changes → upstream RNG state for the next epoch
  drifts. → Mitigation: legacy `cascade_only` had `second_order=False`, so its
  no_grad pass length was tied to `cascade_iters` (50). In the new schema with
  `cascade_1st=True, cascade_2nd=False, second_order=False` this becomes 1 →
  **breaks parity for Setting 2**. Fix: introduce a `legacy_rng_compat` config
  flag (default `True`) that forces the no_grad pass length to `cascade_iters_1`
  when `second_order=False`. Document as `D-blindsight-rng-compat` deviation.

- **Risk 3: AGL has 3 phases (pretrain, training, testing) — D.28
  `train_meta_frozen_in_training` flag interaction with new cascade flags.**
  → Mitigation: explicit table in plan of which phase reads which cascade flag.
  Test against D.28 Phase B 500-seed results.

- **Unknown 1: should setting-5 (cascade 2nd only, 2nd-order on) use the
  Pasquali hidden layer fix from D-blindsight-wager-hidden?** Likely yes (it's
  an architectural fix, not a cascade choice). → Will validate by: running
  setting-5 with both hidden_dim=100 and hidden_dim=0, comparing to paper
  Setting 5 Table 5a: 0.96 ± 0.03 main, 0.87 ± 0.04 wager.

- **Unknown 2: 500 seeds × 2 new settings × 2 experiments = 2000 new runs;
  current Tamia capacity?** Blindsight: ~12h/500 seeds for Setting 6 (paper
  preamble Table 9). Setting 4 + Setting 5 likely similar → ~24h compute on
  Tamia for Blindsight, similar for AGL. → Will plan with sbatch arrays and
  the `aip-gdumas85` shared queue rules (max 3-4 concurrent jobs).

## Verification Plan

How will I know this works?

### Phase α — refactor lands without breaking legacy

- [ ] Unit test: `test_setting_dataclasses.py` — `from_dict` accepts new schema, raises on missing keys, `cascade` legacy property returns `cascade_1st OR cascade_2nd`.
- [ ] Parity test (Blindsight, all 4 legacy cells × 3 seeds): bitwise-equal `losses_1.npy`, `losses_2.npy`, `first_order.pt`, `second_order.pt` against pre-refactor commit.
- [ ] Parity test (AGL, all 4 legacy cells × 3 seeds): same.
- [ ] `ruff check .` clean, `pytest tests/` green.

### Phase β — new settings produce sane output

- [ ] Setting 4 (MAPS, cascade 1st only) on Blindsight: 5-seed pilot → Main Task Acc within [0.93, 1.01] (paper ±2σ).
- [ ] Setting 5 (cascade 2nd only) on Blindsight: 5-seed pilot → Wager Acc within [0.79, 0.95] (paper 0.87 ± 0.04, ±2σ).
- [ ] Setting 4 on AGL high awareness: 5-seed pilot → Main Task Acc within [0.56, 0.76] (paper 0.66 ± 0.05, ±2σ).
- [ ] No NaN/Inf in losses, training curves monotonically converging on log scale.

### Phase γ — full 500-seed runs land in archive

- [ ] Blindsight setting-4 × 500 seeds completes, summary.json present for each.
- [ ] Blindsight setting-5 × 500 seeds completes.
- [ ] AGL setting-4 × 500 seeds completes (3-phase pipeline, all phases logged).
- [ ] AGL setting-5 × 500 seeds completes.
- [ ] Aggregate Z-scores via `aggregate_perceptual.py` → updated headline table in `experiment_matrix.md`.

### Phase δ — close the loop with the paper

- [ ] Compare Blindsight Setting 4 (this work) vs paper Table 5a Setting 4 (z=9.01) — declare reproduction status (within ±2σ / outside).
- [ ] Compare Setting 5 vs Setting 5.
- [ ] Compare new Setting 6 (refactored code) vs archive Setting 6 (legacy code) — must be numerically equivalent (parity Phase α).
- [ ] Same comparisons for AGL.
- [ ] Update deviations.md D.25 closeout narrative: "Sprint-08 D.31 corrected setting labels; previous 0.94 ± 0.03 reproduction was Setting 6, not Setting 4. Setting 4 results: [new number]."

## Definition of Done

- [ ] All α/β/γ/δ checks pass.
- [ ] `BlindsightSetting`/`AGLSetting` dataclasses use the new 3-bool schema; legacy `.cascade` property preserved as a back-compat alias.
- [ ] `config/experiments/factorial_6cell.yaml` is the source of truth; legacy `factorial_2x2.yaml` retained as a 4-cell filter alias.
- [ ] Parity tests prevent silent regression on the legacy 4 cells.
- [ ] `docs/reproduction/deviations.md` has new entries for the cascade-symmetry fix and the D.25 closeout correction.
- [ ] `docs/reproduction/experiment_matrix.md` Blindsight + AGL sections updated with current 6-cell reproduction status.
- [ ] `docs/sprints/sprint-08-reproduction-perceptual-closeout.md` D.31 section appended with summary, plan link, outcome table.
- [ ] Output archives `knowyourself_outputs_<YYYYMMDD>.tar.gz` regenerated with 6 cells × 500 seeds × 2 experiments. Previous archive retained for audit.
- [ ] Reply drafted to N. Kastel with the corrected setting structure, the new headline numbers, and an honest note about the previous mislabelling of `both` as Setting 4.
- [ ] `verification-before-completion` checklist run.

---

## Execution discipline

Per `feedback_microsprint_execution_discipline.md` and
`feedback_plan_execution_discipline.md`: this plan is executed one phase at a
time (α → β → γ → δ), each phase Orient→Do→Verify→Report→Commit→Wait. No
parallel flanks. No silent chaining.

Per `feedback_verify_against_paper_table.md`: each phase verifies against the
paper table, not against the YAML, before being declared done.

Per `feedback_full_reproduction_commitment.md`: if any sub-step blocks, fragment
further — never narrow scope.
