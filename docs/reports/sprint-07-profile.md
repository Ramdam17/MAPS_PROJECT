# Sprint 07 Phase 2 — Bench + Profile

**Status (2026-04-19):** overnight batch **complete**. Phase 2 DoD met (bench, profile, decision drafted). Phase 3/4 reshape required — see §"Wall-clock shock" below.
**Plan ref:** `docs/plans/plan-20260418-sprint07-sarl-tamia.md` §Phase 2.
**Bench rev:** `ac5d2cc`. **Profile rev:** `eef9041`.

---

## Executive summary

1. **The sprint-07 spec's "~500 fps on GPU" assumption is wrong by 5-30×.** Measured: setting 1 GPU = 85 fps, setting 6 GPU = 13 fps. CPU isn't worse and is actually *faster than GPU* for setting 1 (139 fps CPU vs 85 fps GPU).
2. **No single cell of setting 5-6 fits Tamia's 24 h partition cap at 5 M frames.** Setting 6 extrapolates to ~105 h GPU / ~221 h CPU. Settings 1-4 wall unknown (only 1 and 6 benched) but setting 1 fits comfortably (~10 h CPU).
3. **Phase 3 optim priority needs a reshuffle.** Profile shows `target_wager` = **32 % of setting-1 wall** (plan had it at 1-3 %) and `torch.dropout` inside the cascade-iterated forward = **31 % of setting-6 wall** (not in the plan's optim table).
4. **Decision deferred to user:** horizon reduction vs. checkpoint-and-resume infra vs. partial grid (settings 1-3 only). Array sbatch kept at conservative `--time=11:00:00` to accommodate setting 1 CPU only; settings 4-6 need a different strategy before Phase 4.

---

## Bench results (breakout, seed=42)

Filenames: `outputs/bench/bench-<mode>-s<SETTING>-seed42-ac5d2cc.json` (the setting-6 CPU bench predates the filename fix so it's the unsuffixed `bench-cpu_4c-ac5d2cc.json` — content is setting 6).

| Mode | Setting | Frames | Wall (s) | fps | updates/s | VRAM MB | RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| **cpu_4c** | 1 | 500 000 | 3 588.66 | **139.33** | 137.9 | 0 | 1 141 |
| **cpu_4c** | 6 | 50 000 | 7 975.94 | **6.27** | 5.64 | 0 | 1 063 |
| **gpu_full** (h100:4) | 1 | 500 000 | 5 894.39 | **84.83** | 84.0 | 424 | 1 869 |
| **gpu_full** (h100:4) | 6 | 50 000 | 3 787.37 | **13.20** | 11.9 | 319 | 1 719 |

**GPU speedup vs CPU, by setting:**
- Setting 1: 85 / 139 = **0.61×** — GPU *loses* (MinAtar tensors are tiny, kernel-launch-bound).
- Setting 6: 13.2 / 6.27 = **2.11×** — GPU wins, because cascade × 50 iter × batch 32 finally fills the GPU.

VRAM usage ≤ 500 MB / GPU. H100 80 GB is absurdly over-provisioned — but Tamia allocates only by whole node (no MIG), so there's no finer grain.

### Wall-clock shock

Extrapolating to 5 M frames per cell:

| Setting | CPU 4c | GPU 4× h100 |
|---|---:|---:|
| 1 | ~10 h | ~16 h |
| 6 | **~221 h (9 d)** | **~105 h (4.4 d)** |

Settings 2-5 have not been benched. The cost structure between 1 and 6 is driven by `cascade_iterations` — we'd expect settings applying cascade only to first-order (2, 4) to fall between 1 and 6, settings applying cascade to second-order (5) to be closest to 6. A mid-sprint re-bench covering 2/3/4/5 would sharpen planning — scope-add if accepted.

**None of settings 5-6 fit Tamia's 24 h MaxTime (`_b3` tier).** The current `run_sarl.py` has no mid-run checkpoint → `--requeue` can't resume (the idempotency skip in `sarl_array.sh` only triggers once `metrics.json` exists, i.e. after a full run).

---

## Profile results

Profiles were taken with `python -m cProfile` on CPU (no async GPU noise).

### Setting 1 — 100 k frames CPU / 725 s wall (`prof-1-42-eef9041.txt`)

| tottime (s) | ncalls | function | % of total |
|---:|---:|---|---:|
| 233.2 | 94 999 | **`data.py:93 target_wager`** | **32.1 %** |
| 75.3 | 94 999 | `torch._C._EngineBase.run_backward` | 10.4 % |
| 53.9 | 717 591 | `torch._C._nn.linear` | 7.4 % |
| 47.0 | 94 999 | `trainer.py:74 sarl_update_step` | 6.5 % |
| 46.1 | 12 628 630 | `torch.tensor` | 6.4 % |
| 40.2 | 474 995 | `torch.cat` | 5.5 % |
| 36.0 | 239 197 | `torch.conv2d` | 5.0 % |
| 13.0 | 94 999 | `adam.py:347 _single_tensor_adam` | 1.8 % |
| 11.4 | 717 591 | `torch.relu` | 1.6 % |

`target_wager` is by far the biggest hotspot and is on the plan's Phase 3 optim list (**O-1**) — but the plan estimated 1-3 %. Actual: **32 %**. A correct vectorisation could, naively, return ~30 % wall (setting 1 fps: 139 → ~180-200).

### Setting 6 — 25 k frames CPU / 3 584 s wall (`prof-6-42-eef9041.txt`)

| tottime (s) | ncalls | function | % of total |
|---:|---:|---|---:|
| 1 156.3 | 39 998 | `torch._C._EngineBase.run_backward` | **32.3 %** |
| 1 123.6 | 1 067 850 | **`torch.dropout`** (inside cascade forward) | **31.4 %** |
| 485.2 | 7 745 850 | `torch._C._nn.linear` | 13.5 % |
| 252.3 | 2 226 000 | `torch.conv2d` | 7.0 % |
| 145.2 | 3 293 850 | `cascade.py:30 cascade_update` | 4.1 % |
| 107.0 | 6 678 000 | `torch.relu` | 3.0 % |
| 60.8 | 2 226 000 | `model.py:77 forward` | 1.7 % |
| 47.5 | 19 999 | `target_wager` | 1.3 % |
| 28.5 | 19 999 | `sarl_update_step` | 0.8 % |

**`torch.dropout` at 31 % of wall is new**, not in the Phase-3 optim table. 1 067 850 calls / 25 k frames = 43 dropout calls per frame — consistent with cascade 50 × 2 networks × (1 dropout per forward). Each call is ~1 ms on CPU. If the cascade inner-iteration dropout is load-bearing for parity, it can't be skipped; if it's just a side-effect of reusing the model's training forward path, switching to an inference-path forward (`torch.no_grad()` + `model.eval()`-like, but only for cascade inner iterations) would eliminate most of it — subject to a Tier-3 parity test.

**`cascade_update` itself is only 4 %.** The cascade *scaffolding* is cheap; it's what the scaffolding drags (dropout, dropped-into dense forward) that costs.

### Crosswalk vs Phase-3 optim hypotheses

| # | Piste (plan) | Estimé | Mesuré (setting 1) | Mesuré (setting 6) | Verdict |
|---|---|---|---|---|---|
| **O-1** | `target_wager` vectorisation | 1-3 % | **32 %** | 1.3 % | ✅ **keep, priority ↑ dramatique pour settings rapides** |
| **O-2** | `non_terminal_idx` list-comp → nonzero | < 1 % | (non dans top 40) | (non dans top 40) | ⚠️ **impact invisible, dé-priorisation** |
| **O-3** | `get_state` buffer reuse | 2-5 % | (non dans top 40) | (non dans top 40) | ⚠️ **même constat** |
| **NEW O-4** | `torch.dropout` dans cascade inner iters | — | — | **31 %** | 🆕 **à évaluer — risque parité = inconnu, Tier-3 test obligatoire** |

---

## Decision (P2.6, draft — awaits user)

Plan P2.6 threshold: "MIG ≥ 0.6× GPU full → MIG ; GPU ≥ 1.5× CPU → GPU ; else CPU". Tamia has no MIG, so:

- **Setting 1** (and likely 2-4 by analogy): GPU/CPU = 0.61 → **CPU**.
- **Setting 5-6** (cascade-heavy): GPU/CPU = 2.1 → **GPU** *in principle*, but neither CPU nor GPU fits 24 h/cell with 5 M frames. Requires a strategic decision:
  - **(a) Reduce horizon** to 2-3 M frames (violates paper locked methodology — would need to flag as a deviation).
  - **(b) Implement mid-run checkpoint/resume** in `src/maps/experiments/sarl/training_loop.py` so `--requeue` actually resumes; then a GPU cell can chain across 5-6 × 24 h runs.
  - **(c) Ship settings 1-3 (fast) this sprint; defer 4-6 to a follow-up sprint** with (b) landed.
  - **(d) Re-bench settings 2-5** first; if settings 2-4 also land under 15-20 h/cell, option (c) cleanly covers them too.

**Recommendation:** (d) then (c). One more overnight bench (settings 2/3/4/5 at cpu_4c + gpu_full, 50-100 k frames) tells us exactly which settings fit; then ship those in this sprint, plan (b) for the rest.

---

## Array sbatch calibration (P2.7, conservative)

Without a decision on settings 4-6, `sarl_array.sh` is only safe to launch for settings 1-3 (assumed fast). Current header re-calibrated to:

- `--time=11:00:00` (fits setting 1 CPU 5 M ≈ 10 h with 10 % margin).
- `--mem=4000M` (peak RSS measured 1.14 GB, large margin).
- `--array=0-299%15` (bumped from %10 — small jobs, more concurrency).

For settings 4-6 the array should **not** be run as-is. See decision matrix above.

---

## Open issues

- `235782` (CPU setting 6 bench) ran with the pre-fix `bench_sarl.py` so wrote `bench-cpu_4c-ac5d2cc.json` without the setting/seed suffix. Content is correct (JSON has `"setting": 6`). Leave as-is or rename post-hoc.
- `235767` (stale CPU smoke setting-6 50 k) hit `TIMEOUT` at 1 h as expected; never wrote `metrics.json`; dependent `235770` (GPU smoke) was cancelled as planned.
- GPU H100 nodes are under-used: ~400 MB VRAM on an 80 GB card. Future sprint: pack multiple cells per node (`CUDA_VISIBLE_DEVICES`-based), cutting the allocation cost ≈ 4×.
- Settings 2-5 not benched — see Decision (d).

## Bench-to-sprint links

- Phase 3 (optims): now clearly priority-ordered by measured impact. `target_wager` first, then the new `dropout`-in-cascade question. `non_terminal_idx` and `get_state` de-prioritised.
- Phase 4 (array): cannot proceed blindly until the horizon/checkpoint decision is made.
- Phase 5 (report): the "we reproduced" conclusion depends on Phase 4 path; nothing to write yet.
