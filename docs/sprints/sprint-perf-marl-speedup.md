# Sprint — Perf MARL speedup (MAPPO + MeltingPot)

**Status:** 🟡 in progress (Phases 1+2 shipped, blocked on first Narval run)
**Branch:** `perf/marl-mappo-speedup` (off `refactor/blindsight-train`)
**Owner:** Rémy Ramadour
**Started:** 2026-04-17
**Plan file:** `~/.claude/plans/apparemment-mappo-meltingpot-le-deep-otter.md`

---

## TL;DR (15-second resume)

1. Two commits on `perf/marl-mappo-speedup` — profiling instrumentation + SLURM scripts for Narval.
2. **Next concrete action:** run `profile_setting1.sh` and `profile_setting6.sh` on Narval, paste numbers into `docs/reproduction/profiling_report_20260417.md`, then pick **Phase 3** (rollout threads) **or Phase 4** (`torch.compile` on cascade) based on where wall-clock concentrates.
3. Local macOS profiling is blocked — `meltingpot` + `dmlab2d` have no macOS wheels. The preflight in `_preflight.sh` catches this early.

---

## Context (why this sprint exists)

`MARL/MAPPO-ATTENTIOAN/` is a custom MAPPO (fork of `marlbenchmark/on-policy`) running on MeltingPot 2.0 substrates, with two optional MAPS components:
- 2nd-order metacognitive network (Pasquali & Cleeremans 2010)
- Cascade model — 50 iterations of graded accumulation (McClelland 1989, α=0.02)

The 6-setting factorial (meta × cascade on/off) is what the TMLR paper reports. Current wall-clock is "terrible" on Compute Canada. **Mandate: pure technical optimization, NO semantic change** — `share_policy`, `use_centralized_V`, `cascade_iterations=50`, PPO hyperparameters, all architectures remain frozen. Any change that could perturb learning goes in `docs/reproduction/deviations.md` first.

**Expected outcome:** 3–5× wall-clock reduction per setting without touching the paper's behaviour.

---

## Resume on Narval — exact commands

Assumes you have:
- SSH access to Narval (`narval.computecanada.ca`)
- The repo cloned at `$HOME/projects/def-dumasg/$USER/MAPS_PROJECT` (adjust path if allocation differs)
- `module load StdEnv/2023 python/3.11 cuda/12.2 gcc/12.3` available

### Step 1 — clone branch + build venv (one-time)

```bash
ssh narval
cd ~/projects/def-dumasg/$USER/MAPS_PROJECT      # <-- adapt account path
git fetch origin
git checkout perf/marl-mappo-speedup
git pull --ff-only

module load StdEnv/2023 python/3.11 cuda/12.2 gcc/12.3
# One-time venv build. Adjust extras depending on what you need —
# `marl` is the MARL extra (meltingpot + dmlab2d + torch, Linux-only).
virtualenv .venv
source .venv/bin/activate
pip install -e ".[marl,dev]"
python -c "import meltingpot, dmlab2d, torch; print('deps OK')"
```

### Step 2 — short profiling runs (both settings)

```bash
cd ~/projects/def-dumasg/$USER/MAPS_PROJECT
source .venv/bin/activate

# Setting 1 — baseline (no meta, cascade=1). ~50k env steps.
./MARL/scripts/profiling/profile_setting1.sh

# Setting 6 — worst case (meta=True, cascade1=cascade2=50). ~30k env steps.
./MARL/scripts/profiling/profile_setting6.sh
```

Both scripts run `cProfile.runctx()` around `runner.run()` and also log a per-window timing breakdown at each `log_interval`. Artefacts end up in `MARL/outputs/profiling/`:

- `setting{1,6}_<stamp>.prof` — cProfile binary (open with `snakeviz` or `gprof2dot`)
- `setting{1,6}_<stamp>.log` — stdout with the `[timing, s over window Xs]` lines + top-30 cumulative/tottime

> **If you want to run these on a GPU node rather than the login node**, wrap them in a short sbatch — the login node doesn't have GPUs and `torch.cuda.is_available()` will be False. A quick interactive alternative: `salloc --gres=gpu:1 --mem=16G --time=1:00:00` then run the script.

### Step 3 — fill the profiling report

Open `docs/reproduction/profiling_report_20260417.md` and fill:
- **§3 Per-window timing breakdown** — copy-paste the last `[timing]` window from each log
- **§3 Table** — compute the S6/S1 ratio per section (collect, env_step, insert, compute, train)
- **§4–5** — paste the top-30 cumulative and self-time blocks
- **§6 Decision gate** — pick **(A)**, **(B)**, **(C)**, or **(D)** based on the observed hot section, commit the filled-in report as `docs(reproduction): fill profiling report`

### Step 4 — 6 settings × N seeds (when you're ready for the full reproduction)

```bash
cd ~/projects/def-dumasg/$USER/MAPS_PROJECT
./MARL/sbatch/launch_6_settings.sh TERRITORY_I 3     # 6 × 3 = 18 jobs
squeue -u $USER                                       # monitor
```

Submit only after the decision gate has been acted on — otherwise you're burning A100-hours on a config you're about to change.

---

## What's already done (committed on `perf/marl-mappo-speedup`)

| Commit | Phase | What |
|--------|-------|------|
| `cfeaa97` | 1 + 2 | profiling instrumentation + Narval sbatch |
| `ad2df94` | 1.1 | preflight guard for profiling scripts |

### Phase 1 — Profiling (committed)

- `MARL/MAPPO-ATTENTIOAN/onpolicy/runner/shared/meltingpot_runner.py`
  - `time.perf_counter()` around `collect / env_step / insert / compute / train`
  - Per-window breakdown printed every `log_interval` + surfaced to TB/WandB via `train_infos["timing/*"]`
  - Removed `print("before change ...")` from the hot path
- `MARL/MAPPO-ATTENTIOAN/onpolicy/scripts/train/train_meltingpot.py`
  - `--profile / --profile-out` opt-in flags wrapping `runner.run()` with `cProfile`
  - No runtime cost when flag is off
- `MARL/scripts/profiling/profile_setting{1,6}.sh` — short cProfile runs with shared `_preflight.sh`
- `MARL/scripts/profiling/_preflight.sh` — resolves `.venv/bin/python` or `uv run`, fails fast if `torch/meltingpot/dmlab2d` aren't importable
- `pyproject.toml[dev]` — `snakeviz`, `gprof2dot`, `py-spy`
- `docs/reproduction/profiling_report_20260417.md` — **template to fill**

### Phase 2 — SLURM infra (committed)

- `MARL/sbatch/train_maps_marl.sbatch` — 1×A100-40GB + 16 CPUs + 64G, OMP/MKL=1, stages code to `$SLURM_TMPDIR`, syncs outputs to `$SCRATCH`
- `MARL/sbatch/launch_6_settings.sh` — submits the 6×N_SEEDS factorial

---

## What's left (Phases 3 → 5)

### Phase 3 — `n_rollout_threads` (gated on Phase 1 showing env_step ≥ 40%)

- `MARL/meltingpot.sh` — replace `rollout=1` with `rollout=${SLURM_CPUS_PER_TASK:-8}`
- `MAPPO-ATTENTIOAN/onpolicy/envs/env_wrappers.py` — confirm `SubprocVecEnv` is auto-selected when `n_rollout_threads > 1`
- Document in `docs/reproduction/deviations.md` as "purely computational — changes effective batch, not model behaviour"

### Phase 4 — `torch.compile` on cascade (gated on Phase 1 showing train ≥ 40%)

- `MAPPO-ATTENTIOAN/onpolicy/algorithms/utils/rnn.py` — wrap `RNNLayer.forward` with `@torch.compile(mode="reduce-overhead")`
- `MAPPO-ATTENTIOAN/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py:107-122` — cascade loop
- `MAPPO-ATTENTIOAN/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic_meta.py:193-194` — 2nd-order cascade loop
- **Mandatory**: `tests/numerical/test_cascade_equivalence.py` (to create) — `atol=1e-5`, roll back if divergence

### Phase 5 — GPU hygiene / mixed precision (optional, post-Phase 4)

- Hunt `.item() / .numpy() / .cpu()` in the hot path
- Stack agents in the batch dim for a single GPU forward per step (sémantique inchangée)
- `torch.cuda.amp.autocast` + `GradScaler` — discuss with Rémy first (stability risk on 50-iter cascade)

### Phase 1d — deferred, do it only if needed

Adding `time.perf_counter()` granularity inside `r_mappo.ppo_update()` to separate forward-cascade / backward / optimizer-step — only worth it if the Phase 1 report shows `train` is the dominant section. Skip otherwise.

---

## Known pitfalls (not blockers, but don't be surprised)

### Pre-existing bugs in the inherited MARL code — flagged, NOT touched

These are listed in `docs/reproduction/profiling_report_20260417.md §8` and are out of scope for this sprint. Address them in a follow-up:

1. `meltingpot_runner.py:61` — `random.ramdom()` typo → silent `AttributeError`
2. `MARL/meltingpot.sh:360` — broken shell: `if ["$substrate" = ... | ...]; then` (missing spaces, `|` vs `||`)
3. `MARL/meltingpot.sh:23` — hardcoded `/home/juan-david-vargas-mazuera/...`
4. `MARL/meltingpot.sh` — README args 11/12 (meta/cascade) never consumed by the script
5. `train_meltingpot.py:105,110` — hardcoded `'logs/'` (flagged by config-first hook)

### Platform constraints

- **macOS is a dead end for MARL.** `meltingpot`/`dmlab2d` have no macOS wheels. The preflight catches this — don't waste time fighting it.
- `wandb` is set to `False` in the sbatch. Re-enable only if you actually want to watch a run live.
- The sbatch default `--account=rrg-dumasg-ad` is a **placeholder**. Confirm the real RAPI code with Guillaume before submitting — otherwise jobs reject.

### Git discipline

- Commit per phase (don't batch Phase 3 + Phase 4 in the same commit — makes reverts painful)
- Any change that could shift behaviour → `docs/reproduction/deviations.md` entry first, code second
- Do NOT merge to `refactor/blindsight-train` until Tests 1–4 below pass

---

## Verification (Definition of Done)

The sprint is done when all four pass — in order.

### Test 1 — Numerical regression (blocking, Phase 4 only)

```bash
cd MARL/MAPPO-ATTENTIOAN
uv run pytest tests/numerical/test_cascade_equivalence.py -v
```

Compares cascade forward before/after `torch.compile` at `atol=rtol=1e-5`. If it fails, revert the compile wrapping.

### Test 2 — 100-step reward match (deterministic)

```bash
./MARL/meltingpot.sh TERRITORY_I 42 LSTM 100 1 1 META ADAM 42 1 True 0 --num_env_steps 100 --deterministic
```

The reward trajectory on a fixed seed must be unchanged vs. the `main` baseline.

### Test 3 — Throughput (the whole point)

Same setting 6 config on Narval, 10 min wall-clock, **before** vs **after**. Compare `steps/second` logged by `meltingpot_runner.py`. Consigner dans `docs/reproduction/profiling_report_20260417.md §6`.

### Test 4 — Paper reproduction

Run `./MARL/sbatch/launch_6_settings.sh TERRITORY_I 3` (and equivalents for other substrates). Compare z-scores against the table in `docs/reproduction/experiment_matrix.md` — must stay within the paper's error bars.

---

## Decisions still open (ask Rémy before blocking on them)

1. **SLURM account** — placeholder `rrg-dumasg-ad` in `train_maps_marl.sbatch:17`. What's the real RAPI code on Narval? (check with Guillaume)
2. **`n_rollout_threads`** — OK to document as a "purely computational deviation" in `docs/reproduction/deviations.md`? (it changes effective batch seen by PPO)
3. **Mixed precision (Phase 5)** — 50-iter cascade may accumulate numerical error in fp16. Skip unless A100 hours become the bottleneck.
4. **`torch.compile` version** — confirm `.venv` has `torch>=2.0` on Narval (it does on macOS: `torch 2.11.0` per local check).

---

## File map (where things live)

### Touched by this sprint
- `MARL/MAPPO-ATTENTIOAN/onpolicy/runner/shared/meltingpot_runner.py` — timing instrumentation
- `MARL/MAPPO-ATTENTIOAN/onpolicy/scripts/train/train_meltingpot.py` — `--profile` flag
- `pyproject.toml` — dev deps
- `MARL/sbatch/{train_maps_marl.sbatch,launch_6_settings.sh}` — new
- `MARL/scripts/profiling/{_preflight.sh,profile_setting1.sh,profile_setting6.sh}` — new
- `docs/reproduction/profiling_report_20260417.md` — template, **to fill**

### Referenced but not yet touched (Phases 3/4)
- `MARL/meltingpot.sh` — rollout param bump (Phase 3)
- `MARL/MAPPO-ATTENTIOAN/onpolicy/algorithms/utils/rnn.py` — `torch.compile` target (Phase 4)
- `MARL/MAPPO-ATTENTIOAN/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py:107-122` — cascade loop
- `MARL/MAPPO-ATTENTIOAN/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic_meta.py:193-194` — 2nd-order cascade loop
- `MARL/MAPPO-ATTENTIOAN/onpolicy/envs/env_wrappers.py` — SubprocVecEnv check
- `tests/numerical/test_cascade_equivalence.py` — **to create** in Phase 4

### Not in scope (frozen)
- Architecture, PPO hyperparameters, cascade/meta semantics — paper-locked (`config/maps.yaml`)
- Pre-existing bugs listed under "Known pitfalls" — separate sprint

---

## Ordre d'exécution recommandé (weekend)

```
[you are here]
      │
      ▼
Step 1 — clone/venv on Narval
      │
      ▼
Step 2 — profile_setting1.sh + profile_setting6.sh
      │
      ▼
Step 3 — fill profiling_report_20260417.md, commit
      │
      ▼
Decision gate:
  ├─ (A) train dominates   → Phase 4 torch.compile
  ├─ (B) env_step dominates → Phase 3 rollout threads
  ├─ (C) insert dominates  → Phase 3.5 obs-dtype fix
  └─ (D) mixed            → Phase 3 + Phase 4 in parallel
      │
      ▼
Implement chosen phase on perf/marl-mappo-speedup
      │
      ▼
Tests 1 → 2 → 3 → 4
      │
      ▼
Merge to refactor/blindsight-train (after Rémy validates)
```

Each step produces a git-committable artefact. No rebase/squash until Test 4 has passed.
