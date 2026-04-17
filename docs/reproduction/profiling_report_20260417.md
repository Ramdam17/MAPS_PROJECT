# Profiling Report — MAPPO + MeltingPot

**Date:** 2026-04-17 (template — fill in after first run)
**Branch:** `perf/marl-mappo-speedup`
**Goal:** localize the real bottleneck in the MAPS/MARL training loop before
applying Phase 3+ optimizations.

---

## 1. Context

The MARL codebase (`MARL/MAPPO-ATTENTIOAN/`) is a custom MAPPO
implementation (no Ray RLlib). Training is reportedly "terribly slow" on
Compute Canada (Narval). The plan
(`~/.claude/plans/apparemment-mappo-meltingpot-le-deep-otter.md`) calls for
evidence-based profiling of **setting 1** (baseline) and **setting 6**
(worst case: meta=True + cascade_iterations1=50 + cascade_iterations2=50)
before any optimization is applied.

---

## 2. Runs captured

| Run | Setting | Substrate | Agents | Episodes | Env steps | Wall-clock | Command |
|----:|--------:|-----------|-------:|---------:|----------:|-----------:|---------|
| 1   | 1       | territory__inside_out | 5 | 5 | 50 000 | _TBD_ | `scripts/profiling/profile_setting1.sh` |
| 2   | 6       | territory__inside_out | 5 | 3 | 30 000 | _TBD_ | `scripts/profiling/profile_setting6.sh` |

Artifacts live in `outputs/profiling/`.

---

## 3. Per-window timing breakdown (from `meltingpot_runner.py` instrumentation)

Filled from the `[timing, s over window Xs]` lines logged at every
`log_interval`. Use the **last window** of the run (warm cache).

### Setting 1
```
collect  : __.__ s (__%)
env_step : __.__ s (__%)
insert   : __.__ s (__%)
compute  : __.__ s (__%)
train    : __.__ s (__%)
```

### Setting 6
```
collect  : __.__ s (__%)
env_step : __.__ s (__%)
insert   : __.__ s (__%)
compute  : __.__ s (__%)
train    : __.__ s (__%)
```

### Setting 6 vs setting 1 — per-section ratio

| Section   | S1 (s/ep) | S6 (s/ep) | S6/S1 | Expected |
|-----------|----------:|----------:|------:|----------|
| collect   |           |           |       | ~50×     |
| env_step  |           |           |       | ~1× (no model dep) |
| insert    |           |           |       | ~1×      |
| compute   |           |           |       | ~1×      |
| train     |           |           |       | ~50–100× |

*If `env_step` dominates even in setting 1, the cascade/meta is not the
bottleneck. Reorient Phase 3 (rollout parallelism) before Phase 4
(torch.compile).*

---

## 4. cProfile top 30 — setting 6, cumulative

```
(pasted from outputs/profiling/setting6_<stamp>.log)

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   ...
```

### Findings

- [ ] Where does `cumtime` concentrate? (expected: cascade loop in `RNNLayer.forward`)
- [ ] Any surprise hot function? (numpy reshape, pickle, `_t2n`, etc.)
- [ ] Any I/O or sleep? (check for `time.sleep`, `imageio.write`, `torch.save` in hot path)

---

## 5. cProfile top 30 — setting 6, self-time

```
(pasted from outputs/profiling/setting6_<stamp>.log, second block)
```

### Findings

- [ ]
- [ ]

---

## 6. Decision gate — which phase next?

Based on the above, one of:

- **(A) Cascade/meta dominates (≥40% of train_infos[train])** → Phase 4
  `torch.compile` on `RNNLayer.forward` and `SecondOrderNetwork.forward`.
  Expected gain: 3–5× on setting 6.
- **(B) env_step dominates (≥40% overall)** → Phase 3 `n_rollout_threads`
  bump from 1 → 8–16 (with OMP_NUM_THREADS=1). Expected gain: ~N× where N
  is the SubprocVecEnv count, capped by CPU-count.
- **(C) insert/numpy overhead dominates** → inline the per-step
  `np.squeeze` / `np.transpose` and cast obs to uint8 earlier (no
  semantic change).
- **(D) mixed: both cascade and env_step are costly** → pursue Phase 3
  and Phase 4 in parallel; they do not conflict.

**Decision taken:** _TBD_

---

## 7. Deviations recorded

List any knob changed here that differs from the paper's original
configuration, so `docs/reproduction/deviations.md` can be kept in sync.

- `n_rollout_threads`: 1 → N (purely computational — changes effective
  batch seen by PPO but not model behaviour). Document in deviations.md.
- `torch.compile` wrapping: confirmed numerically equivalent by
  `tests/numerical/test_cascade_equivalence.py` at atol=1e-5.

---

## 8. Pre-existing issues surfaced during the audit

Not perf-related, flagged for Rémy to address in a separate sprint:

1. **`meltingpot_runner.py:61`** — `random.ramdom()` (typo for `random.random()`).
   Crashes at first call if episode_length > 5000 randomly triggers; otherwise
   the bug is latent because the `while` loop always runs `.ramdom()` eagerly —
   likely causing a silent `AttributeError` somewhere.
2. **`MARL/meltingpot.sh:360`** — `if ["$substrate" = ... | ...]; then` is
   broken shell (missing spaces, `|` instead of `||`). The branch probably
   never fires.
3. **`MARL/meltingpot.sh:23`** — `general_dir="/home/juan-david-vargas-mazuera/..."`
   hardcoded to the previous maintainer's path on Compute Canada. Must be
   parameterised via env var before any Narval run.
4. **`MARL/meltingpot.sh`** — args 11 (`meta`) and 12 (`cascade`) shown in the
   README are never consumed by the bash script. Only `--setting` (arg 10) is
   passed to train_meltingpot.py, and train_meltingpot.py overrides meta/cascade
   from `setting` anyway. The README example is misleading.
5. **`train_meltingpot.py:105,110`** — `'logs/'` hardcoded (flagged by
   config-first auditor hook). Should move to `paths.yaml`.
