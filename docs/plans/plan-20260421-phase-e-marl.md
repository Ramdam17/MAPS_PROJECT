# Phase E — MARL MeltingPot reproduction (18 sub-phases)

**Author :** Rémy Ramadour + Claude, 2026-04-21.
**Sprint target :** Sprint 09 (E.1–E.18).
**Commitment :** full paper-faithful reproduction, no pivoting to lesser scope.

---

## 1. Context & scope

Paper §A.4 + §B.4 + Table 12 describe MAPS on MeltingPot 2.0 via MAPPO (Yu et
al. 2022) with 4 environments :
1. Commons Harvest Closed (6 agents)
2. Commons Harvest Partnership (4 agents)
3. Chemistry Three Metabolic Cycles with Plentiful Distractors (8 agents)
4. Territory Inside Out (5 agents)

Paper §B.4 preamble explicitly states *"MAPS not implemented fully, only with
simple 2nd order network with no cascade model due to limitations with
computational resources."* → Port target is paper-declared simple-2nd-order
MAPPO + no cascade (D-marl-cascade-not-implemented, already in deviations.md).

**Target paper numbers (Table 7, p. 17 — z-scores vs baseline) :**
- Commons Harvest Closed : z ≈ (to-be-extracted from Table 7)
- Commons Harvest Partnership : z ≈ ...
- Chemistry : z ≈ ...
- Territory Inside Out : z ≈ ...

**Success criteria (DoD) :**
- All 4 envs reproduce within ±2σ of paper Table 7 z-scores.
- 3 seeds per env × 6 settings (or simplified to 2 settings per paper
  preamble : baseline + MAPS-simple).
- All architectural deviations (D-marl-*) either resolved or explicitly
  documented with evidence.
- Parity tests (forward, buffer, update) bit-exact or epsilon vs vendored
  student reference.
- Resolution doc + `deviations.md §B.11` fully closed.

**Blockers to expect :**
- `dmlab2d` + `meltingpot` install on DRAC Tamia (Linux-only native deps).
- Compute cost : paper says 16h/seed on A100. For 3 × 4 = 12 seeds : ~192 GPU
  hours for MAPS run × 6 settings — could be 1000+ GPU hours total. Mitigated
  by (a) paper only runs 2 useful settings, (b) tamia has H100/H200 (faster).

**Student code source :** `MARL/MAPPO-ATTENTION/` was deleted from repo tree
in Sprint 04b/07. Must be restored from git history for reference.

---

## 2. 18 sub-phases

### Sprint 09a — Audit + environment (E.1-E.6)

#### E.1 — Restore student MARL code from git history (~1 h)
**Goal :** snapshot `MARL/MAPPO-ATTENTION/` from the last commit before its
deletion into `external/paper_reference/marl_tmlr/`. Read-only.

- Find the SHA where `MARL/` was last present (pre-Sprint 04b).
- `git checkout <SHA> -- MARL/MAPPO-ATTENTION`
- Move to `external/paper_reference/marl_tmlr/` (following Blindsight/AGL
  naming pattern).
- Add `external/paper_reference/marl_tmlr/.gitignore` for pycache.
- Commit.

#### E.2 — Audit student MAPPO core architecture (~1 day)
**Goal :** document `onpolicy/algorithms/r_mappo/algorithm/` :
- `r_actor_critic.py` (actor + critic networks)
- `r_actor_critic_meta.py` (MAPS 2nd-order additions — comparator, wager)
- `rMAPPOPolicy.py` (policy wrapper)
- `r_mappo.py` (trainer, PPO loss, advantage computation)

Produce `docs/reviews/marl-architecture.md` with component-by-component map.

#### E.3 — Audit MeltingPot env wiring + reward/termination (~1/2 day)
**Goal :** document `onpolicy/envs/meltingpot/` :
- Observation shape per env (11×11 RGB + agent-specific state).
- Reward scaling per env.
- Termination conditions.
- Positional encoding (sinusoidal, paper §2.2) + GRU.

`docs/reviews/marl-env.md`.

#### E.4 — Audit MAPS additions (comparator + wager + loss) (~1/2 day)
**Goal :** cross-reference `r_actor_critic_meta.py` with paper §2.2 + Fig.4 +
eq. 1-6. Identify deviations from SARL's `SarlSecondOrderNetwork`.

Key finding from B.11 audit (already done) : MARL has a **true
`comparison_layer` linear+ReLU+Dropout** (not tied-weights like SARL).
Document this.

`docs/reviews/marl-maps-additions.md`.

#### E.5 — Decide on RIM / SCOFF / skill_dynamics / bottom_up extensions (~1 h)
**Goal :** paper Fig.4 shows plain GRU only ; student has RIM + SCOFF + skill
dynamics + bottom-up attention (from the forked MAPPO-ATTENTION repo).

Per B.11 audit decision : **OMIT these extensions** (paper-faithful scope).
Document in deviations.md as `D-marl-attention-extensions` (already exists,
marked "policy decision").

**DoD :** decision formalized, student extensions flagged as "read-only
reference, not ported".

#### E.6 — MeltingPot install verification on DRAC Tamia (~1 day, BLOCKER)
**Goal :** verify `dmlab2d`, `meltingpot`, `ray[default,tune]`, `gymnasium`
all install + import + smoke-test on DRAC Tamia.

- Per `docs/install_linux.md`, follow manual install path.
- Test on CPU first (login node forbidden per memory, use `salloc`).
- Test on GPU (H100/H200).
- `python -c "import dmlab2d; import meltingpot.python.substrate as s; env = s.get_factory('commons_harvest_closed_partnership').build(); print('OK')"`
- Document any DRAC-specific patches.

**DoD :** one substrate builds and steps on DRAC. If blocked, create
`docs/install_marl_drac.md` with workarounds until resolved.

---

### Sprint 09b — Port MAPPO (E.7-E.12)

#### E.7 — Scaffold `src/maps/experiments/marl/` (~1/2 day)
**Goal :** package skeleton mirroring SARL pattern :
- `__init__.py`, `setting.py` (factorial abstraction), `trainer.py`,
  `policy.py`, `env.py`, `buffer.py`, `data.py` as needed.
- `config/training/marl.yaml` + `config/env/marl.yaml` + per-substrate
  overrides.

No logic yet, just structure + dataclasses.

#### E.8 — Port MAPPO policy + critic (~1-2 days)
**Goal :** `maps.experiments.marl.policy.MAPPOPolicy` + `Critic` in pure
PyTorch, bit-compatible with student (weights loadable cross-implementation).

Components to port :
- ConvEncoder (from obs 11×11×RGB → flat features, paper §2.2 Fig.4).
- Positional encoding (sinusoidal).
- GRU (stability, paper §2.2).
- Actor head (categorical action logits).
- Critic head (value).
- MAPS additions : comparator matrix, 2-unit wager head.

Config : `config.model.hidden_size = 100` (paper T.12), not student's 144.

#### E.9 — Port MAPPO training loop (~2 days)
**Goal :** `maps.experiments.marl.trainer.MAPPOTrainer` :
- On-policy rollout collection (env.step × n_agents × n_envs × n_rollout_threads).
- GAE advantage estimation.
- PPO clip loss + value loss + entropy bonus.
- Optimizer update with gradient clipping.
- EMA wagering signal (paper eq. 13-14).

Config : paper T.12 values (actor_lr=7e-5, critic_lr=7e-5, entropy=0.01,
clip=0.2, ppo_epoch=15, weight_decay=1e-5, num_env_steps=15e6, optimizer=Adam).

#### E.10 — Port MeltingPot env wrappers (~1 day)
**Goal :** `maps.experiments.marl.env.MeltingPotWrapper` — adapts
`meltingpot.python.substrate` to gymnasium-compatible interface expected by
the trainer. Per-agent observation dict → flat tensor batch.

Per-env configs with correct `num_agents`, obs shape, action space.

#### E.11 — Port reward/termination per env (~1 day)
**Goal :** per-substrate logic. 4 substrates :
- Commons Harvest Closed (6 agents)
- Commons Harvest Partnership (4 agents)
- Chemistry Three Metabolic Cycles (8 agents)
- Territory Inside Out (5 agents)

Config per-env YAML under `config/env/marl/<substrate>.yaml`.

#### E.12 — CLI scripts + slurm templates (~1 day)
**Goal :** `scripts/run_marl.py` (typer) + `scripts/slurm/marl_array.sh`
(GPU, `--gres=gpu:h100:1` or `h200:1`, 16-24h wall per seed).

Include `--substrate`, `--setting`, `--seed`, `--output-dir`, `--num-env-steps`,
`--resume`, `--resume-from` flags.

---

### Sprint 09c — Parity + smoke + full run (E.13-E.18)

#### E.13 — Parity tier-1 : MAPPO forward (~1 day)
**Goal :** per-step forward pass bit-exact vs student. Seeded init, single
obs, one action, one value. atol=1e-6.

Mirror `tests/parity/sarl/test_tier1_forward.py`.

#### E.14 — Parity tier-2 : MAPPO buffer/rollout (~1 day)
**Goal :** 10-step rollout bit-exact : obs, actions, rewards, returns,
advantages. atol=1e-5.

Mirror `tests/parity/sarl/test_tier2_buffer.py`.

#### E.15 — Parity tier-3 : MAPPO update (~1 day)
**Goal :** full PPO update step (clip loss + value loss + entropy + backward +
optimizer step) bit-exact. atol=1e-5.

Mirror `tests/parity/sarl/test_tier3_update.py`.

#### E.16 — Smoke run on DRAC GPU (~1/2 day)
**Goal :** 1 substrate × 1 seed × 100k env steps on H100 via sbatch. Verify
pipeline end-to-end. Expected wall : < 1 h.

#### E.17 — Full run : 3 seeds × 4 envs × 6 settings × 15M steps (~1-2 weeks wall)
**Goal :** paper Table 12 grid on DRAC H100. Chained sbatch, respect lab
queue discipline.

Per paper §B.4 preamble, only "simple 2nd-order" variant is meaningful →
effectively 2 settings :
- Setting 1 : baseline MAPPO.
- Setting 3 or 6 : MAPS (2nd-order on, cascade off per D-marl-cascade).

So 3 × 4 × 2 = **24 runs × 16h = ~400 GPU-hours**. Conservative : budget 500.

All config aligned to paper T.12 (not student config): hidden=100, actor_lr=7e-5,
entropy=0.01, num_env_steps=15e6.

#### E.18 — Aggregation + paper Table 7 comparison + closeout (~1 day)
**Goal :** compute z-scores per env, compare to paper Table 7, write
`docs/reviews/rg-marl-resolution.md`, update `deviations.md §B.11` to all ✅,
Phase E closeout commit.

Parallel ablations (if compute allows) : test hidden=100 vs student's 144,
entropy=0.01 vs 0.004.

---

## 3. Risk register

| Risk | Likelihood | Impact | Mitigation |
|:--|:--:|:--:|:--|
| `dmlab2d` install fails on DRAC | M | H (blocks everything) | E.6 early. If fails : patch `dmlab2d` build / use older CUDA toolchain / container image |
| `meltingpot` substrate missing on DRAC | L | H | Fallback : install meltingpot from local pip wheel bundled in repo |
| MAPPO port drifts from student numerics | M | M | Parity tests E.13-E.15 catch at each tier |
| RangerVA-analog Adam optimizer misalign | L | L | Paper T.12 says Adam, same as torch default |
| Compute budget exceeded | M | M | Chain sbatch via `--dependency`, use `--preemptable` on tamia if available |
| Port scope creep (RIM/SCOFF temptation) | M | H | E.5 decision lock, reference code is read-only |
| Paper Table 7 numbers not reproducible | L | L | If so, ablate + document as `D-marl-reproduction-gap` new deviation ; but first assume reproducibility |

---

## 4. Effort estimate (realistic)

| Sprint | Sub-phases | Eng days | DRAC GPU hours |
|:--|:--|:--:|:--:|
| 09a Audit + install | E.1–E.6 | 4 | 0-2 (install smoke) |
| 09b Port | E.7–E.12 | 7-10 | 0 (local CPU smoke) |
| 09c Parity + run | E.13–E.18 | 4-6 eng + 2 weeks wall | ~500 GPU-hours |

**Total : ~3 weeks engineering + 2 weeks DRAC compute.**
Commit discipline : each sub-phase atomically committed, as per Phase D.

---

## 5. Dependencies

- Phase E follows Phase D closure (all RG-001/002/003 ✅).
- `torch-optimizer` installed (AGL dep, no MARL impact).
- Need :
  - `ray[default,tune]`, `gymnasium`, `dm-env`, `dm-tree`, `tensorboardX`, `wandb`
    (already in `marl` extras).
  - `dmlab2d` + `meltingpot` manually (see docs/install_linux.md).

---

## 6. Go / no-go gate before E.1

- [ ] Plan v1 validated by Rémy.
- [ ] Lab queue discipline confirmed : 3-4 concurrent jobs, chain via
  `--dependency`, `--mem=Xxxx M` not `XG`.
- [ ] OK to restore deleted `MARL/MAPPO-ATTENTION/` tree into
  `external/paper_reference/marl_tmlr/` (read-only).

Once confirmed, we start E.1.
