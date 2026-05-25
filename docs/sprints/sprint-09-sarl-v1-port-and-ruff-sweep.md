# Sprint 09 — SARL `maps_v1.py` port + ruff sweep + MARL alignment

**Status:** 🔴 not started
**Branch:** `repro/sprint-09-sarl-v1` (to be created off `main` after PR #1 merges)
**Owner:** Rémy Ramadour
**Estimated effort:** ~3-5 days engineering + DRAC compute for v1 Phase F re-runs
**Depends on:** Sprint 08 PR #1 merged (Tamia+Narval reproduction integrated)

---

## Context

Sprint 08's merge PR (`merge/sprint-08-tamia-narval`) brought 121 commits of Tamia + Narval
work into main, including the full SARL port and Phase F.4 SARL+CL curriculum runs. But the
last commit on the merge source (`6c59744 docs(repro): restore maps_v1.py + flag v1/v2
wrong-variant port`) revealed a **critical architectural mismatch** : the entire SARL
reproduction path targets `maps_v2.py`, while the paper's Table 6 numbers come from
`maps_v1.py` invoked by `SARL_Training_Standard.sh base=2000000 -ema 25`.

The Sprint-04b TD-008 audit that justified using v2 was a token-presence comparison
("v2 adds only `count_parameters` and `curriculum`") — not a structural diff. The two
variants differ on load-bearing pieces : Q-head input dim (1024 vs 128), reconstruction
decoder (dedicated 131K-param layer vs tied weights), `comparison_layer` (active 1M-param
layer vs commented-out), cascade target (1024-dim Output vs 128-dim Hidden), and training
budget (2M frames at α=0.25 vs 500k at α≤0.45).

This sprint exists to fix that mismatch and produce paper-faithful SARL numbers.

In addition, the Sprint-08 merge inherited ~135 ruff issues (NPY002, B905, RUF005, RUF012,
RUF059, B017, N811) from the wip/centralize branch (ruff was pinned at `v0.15.11` in
`.pre-commit-config.yaml` while the wip commits predated newer rules). The merge PR was
pushed with `--no-verify` to unblock the integration. Sprint 09 closes that debt as part
of the same branch.

Finally, the MARL port (E.1-E.18, fully ported on the merge) has 4 unresolved deviations
flagged in `experiment_matrix.md` and `paper_vs_code_audit.md §B.11` : `D-marl-num-env-steps`,
`D-marl-hidden-size`, `D-marl-actor-lr`, `D-marl-entropy-coef`. Phase E.17 was scoped to
address these but completed only the smoke test (E.16). The full audit + alignment is
included here.

## Open items (entry conditions)

1. **D-sarl-wrong-variant** (severity CRITICAL, in `docs/reproduction/deviations.md` §B.7)
2. **TD-008** retracted (in `docs/TODO.md`, "near-identical, v2 adds only count_parameters")
3. **Ruff debt** : 87 unfixable + 48 already auto-fixed (during merge attempts) — needs sweep
4. **MARL deviations** : D-marl-num-env-steps, D-marl-hidden-size, D-marl-actor-lr,
   D-marl-entropy-coef — flagged in Phase B.11 audit, partially scoped in E.17
5. **MARL Setting 7 unblock** : pending Juan/Guillaume clarification on IMPALA+CPC training
   code (Agapiou 2023, MeltingPot baseline). Not actionable in Sprint-09 unless Juan replies.

## Goals (DoD)

### Phase 9.1 — Port `maps_v1.py` (Days 1-2)

- [ ] Restore `external/paper_reference/sarl_maps_v1.py` from Juan's fork
  (`SARL/MinAtar/examples/maps_v1.py`, pre-`8aa1138`).
- [ ] Implement `src/maps/experiments/sarl/model_v1.py` with `SarlQNetworkV1` and
  `SarlSecondOrderNetworkV1` matching the v1 structural diffs documented in
  `docs/reproduction/sarl-v1-vs-v2.md`.
- [ ] Add bit-exact parity test `tests/parity/sarl/test_sarl_v1_parity.py` against the
  restored reference under seeded inputs.
- [ ] Add `D-sarl-wrong-variant` resolution entry (mirror format of D-sarl-recon-bias).
- [ ] Retract TD-008 in `docs/TODO.md` with reference to this sprint.

### Phase 9.2 — Wire v1 into the training stack (Day 3)

- [ ] Add config toggle `training.sarl_model_variant ∈ {"v1", "v2"}` in `config/training/sarl.yaml`,
  default `"v1"` (paper-faithful).
- [ ] Plumb the variant through `MAPS.SarlTrainer.build()` so the right network architecture is
  instantiated.
- [ ] Update `scripts/slurm/sarl_array.sh` to default to 2M frames and α=0.25 when variant=v1
  (paper Table 6 reproduction target).
- [ ] Smoke run `bench_sarl.sh` on M-Max or Tamia (1 game, 100k frames) to confirm v1 forward
  pass + gradient flow.

### Phase 9.3 — Phase F re-submission on Narval (Days 3-5)

- [ ] Re-submit Settings 1-7 × 5 games × 3 seeds with v1 at 2M frames on Narval
  (`scripts/slurm/sarl_array.sh` updated).
- [ ] Archive current Phase F.4 v2 results under `docs/reports/phase-F-v2-reference/` with a
  prominent README noting they are NOT the paper-faithful target.
- [ ] Run aggregation `scripts/aggregate_sarl.py --variant v1` and compare against
  `juan_sarl_results.csv` (Natalie's TMLR-deadline extraction).
- [ ] If v1 numbers diverge from paper, audit `maps_v1.py` once more for hidden
  hyperparameter dependencies and document in deviations.md.

### Phase 9.4 — Ruff sweep (Day 4, parallel)

- [ ] Run `uv run ruff check --fix .` on the full tree and inspect the auto-fix diff.
- [ ] Run `uv run ruff check --fix --unsafe-fixes .` on the remaining 38 issues and inspect.
- [ ] Manually fix the residual 49 issues (mostly RUF059 unused vars, B017 blind assertions,
  N811 constant-name imports).
- [ ] Re-run `uv run pytest -m "not slow and not gpu and not linux_only" -q` and confirm
  green.
- [ ] Re-run `pre-commit run --all-files` and confirm green.
- [ ] Squash-commit `chore(ruff): sweep across Phase D-F merged code`.

### Phase 9.5 — MARL deviation alignment (Day 5)

- [ ] Audit Phase E.17 carryover in `docs/reproduction/paper_vs_code_audit.md §B.11`.
- [ ] Align `config/training/marl.yaml` hidden_size to 100 (paper), document override path
  to 144 (current port).
- [ ] Align actor_lr to 7e-5 vs shell-passed 2e-5 (decide which is canonical via E.17 audit).
- [ ] Align entropy_coef to 0.01 vs shell-passed 0.004.
- [ ] Align num_env_steps to 300k (text) or 15M (config) — pick based on smoke test wall-clock
  budget.
- [ ] Re-run 3-seed smoke arrays on Tamia to confirm no regression on E.16 baseline.

### Phase 9.6 — MARL Setting 7 (if Juan replies)

- [ ] If Juan/Guillaume thread on Setting 7 (IMPALA + CPC training code) resolves, fold the
  external code into `external/paper_reference/marl_setting_7.py` and scope a separate
  E.18+ port. Otherwise mark `D-marl-setting-7` as "blocked external".

## Verification (end-to-end)

```bash
# 1. v1 parity green
uv run pytest tests/parity/sarl/test_sarl_v1_parity.py -v

# 2. v2 archive verified
ls docs/reports/phase-F-v2-reference/
test -s docs/reports/phase-F-v2-reference/README.md

# 3. v1 paper-faithful numbers within ±2σ of Juan's CSV
uv run python scripts/aggregate_sarl.py --variant v1 \
  --compare juan_sarl_results.csv --tolerance 2sigma

# 4. ruff clean
uv run ruff check . && uv run ruff format --check .

# 5. pre-commit clean on all touched files
uv run pre-commit run --all-files

# 6. MARL deviations resolved
grep -c "🆘\|🚨" docs/reproduction/deviations.md   # should drop vs Sprint-08 baseline
```

## Non-goals (out of scope)

- **METTA-AI** integration (separate environment, `external/METTA`)
- **SARL+CL Figure 7** re-runs (already submitted on Narval with v2 — will be re-done in
  Sprint-10 after v1 is validated)
- **Blindsight / AGL** further work — Sprint-08 closed those (RG-002, RG-003).
- **MeltingPot Setting 7 IMPALA training** — blocked external.

## Risks

1. **v1 numbers also fail to reproduce paper.** Then we have a deeper issue : either Juan's
   public fork is incomplete, or the paper itself used unpublished hyperparameters. Mitigation:
   review `juan_sarl_results.csv` to see whether even Juan reproduces Table 6 numbers.
2. **2M-frame Narval runs exceed allocation.** 7 settings × 5 games × 3 seeds × 2M frames is
   ~63h GPU. Mitigation: stagger over 3 days, or reduce to 2 seeds with documented variance
   note.
3. **Ruff sweep collides with v1 port commits.** Mitigation: do ruff sweep first on a separate
   branch, merge, then start v1 work on top.

## File touchpoints (planned)

- `src/maps/experiments/sarl/model_v1.py` — **new**
- `src/maps/experiments/sarl/__init__.py` — add v1 exports
- `src/maps/experiments/sarl/training_loop.py` — variant routing
- `config/training/sarl.yaml` — `sarl_model_variant` toggle, `num_frames`/`alpha` defaults
- `scripts/slurm/sarl_array.sh` — variant-aware batch params
- `tests/parity/sarl/test_sarl_v1_parity.py` — **new**
- `external/paper_reference/sarl_maps_v1.py` — **restored from Juan fork**
- `docs/reproduction/deviations.md` — resolve D-sarl-wrong-variant
- `docs/TODO.md` — retract TD-008
- `docs/reports/phase-F-v2-reference/README.md` — **new**, archive v2 runs
- Ruff sweep: ~50 files in `src/`, `scripts/`, `tests/` (auto-fix + manual)
- `config/training/marl.yaml` — hidden_size, actor_lr, entropy_coef, num_env_steps

## References

- `docs/reproduction/sarl-v1-vs-v2.md` — full v1↔v2 architectural diff (created `6c59744`)
- `docs/reproduction/deviations.md §B.7 D-sarl-wrong-variant` — entry to resolve
- `docs/reports/natalie-update-20260520.md` — Natalie's audit response draft
- `external/paper_reference/sarl_maps.py` — current v2 reference (vendored from Juan fork)
- `tests/parity/sarl/_reference_sarl.py` — Sprint-04b reference (v2-targeting)
