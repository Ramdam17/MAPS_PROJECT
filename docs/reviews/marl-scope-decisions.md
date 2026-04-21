# MARL port scope decisions (E.5 lock)

**Phase E.5 — formalize decisions taken during E.2/E.3/E.4 audits.**
**Reviewer :** Rémy Ramadour + Claude, 2026-04-21.
**Purpose :** freeze the port scope **before** any code is written (E.7+) to
prevent later scope creep.

---

## (a) Decision principles (carried from Phase D)

1. **Paper = source of truth.** Student code defers to paper prose/equations
   when it diverges. Exceptions (e.g. Blindsight hidden_dim=40, AGL n_epochs=30)
   are documented in `deviations.md` with ablation evidence.
2. **Student code > paper tables** when the two disagree on numerical config
   (observed D.25 + D.28.h : student code produced the paper numbers, tables
   are lossy summaries).
3. **Dead code is dropped.** Unused parameters, class fields, call-sites
   flagged during audit do not enter the port.
4. **Extensions outside paper Fig.4 are OMITTED** (RIM, SCOFF, MPE, SMAC,
   football, hanabi, set_transformer, skill_dynamics, bottom-up, mat/hatrpo).
5. **Single canonical runner path** : separated only (shared is broken).

---

## (b) Included in port ✅

| Component | Source file | Port target | Rationale |
|:--|:--|:--|:--|
| `CNNBase` + `CNNLayer` + `Encoder` | `utils/cnn.py` | `src/maps/experiments/marl/encoder.py` | Paper Fig.4 ConvEncoder |
| `RNNLayer` (baseline) | `utils/rnn.py` | `src/maps/experiments/marl/rnn.py::RNNLayer` | Paper Fig.4 GRU |
| `RNNLayer_Meta` (keep GRU+LayerNorm+cascade1) | `utils/rnn_meta.py:1-96` | `src/maps/experiments/marl/rnn.py::RNNLayerMeta` | Paper Fig.4 GRU + eq.6 cascade |
| `R_Actor`, `R_Critic` | `r_mappo/algorithm/r_actor_critic.py` | `policy.py::MAPPOActor`, `MAPPOCritic` | Paper Fig.4 baseline |
| `R_Actor_Meta`, `R_Critic_Meta` | `r_mappo/algorithm/r_actor_critic_meta.py` | `policy.py::MAPSActor`, `MAPSCritic` | Paper Fig.4 with wager |
| `SecondOrderNetwork` (MARL variant, 2 units, comparison_layer) | `r_actor_critic_meta.py:17-42` | `policy.py::MarlSecondOrderNetwork` | Paper §2.2 eq.1-3 |
| `R_MAPPOPolicy` wrapper | `rMAPPOPolicy.py` | `policy.py::MAPPOPolicy` | 4-network manager |
| `R_MAPPO` trainer (PPO clip + value + entropy + meta BCE) | `r_mappo.py` | `trainer.py::MAPPOTrainer` | Paper §2.2 eq.5 + standard PPO |
| `MeltingPotEnv` + `DownSamplingSubstrateWrapper` + `env_creator` | `envs/meltingpot/MeltingPot_Env.py:124-360` | `env.py` | 4 target substrates |
| `MeltingpotRunner` (separated variant) | `runner/separated/meltingpot_runner.py` | `runner.py` (inline helper methods merged into trainer) | EMA wager + PPO epoch orchestration |
| Optimizers : Adam + RangerVA | `rMAPPOPolicy.py:51-96` | 2 branches only | Paper T.12 says Adam ; fallback RangerVA |
| Action layer | `utils/act.py::ACTLayer` | `src/maps/experiments/marl/act.py` | Discrete / Multi-discrete action heads |
| `ValueNorm` (optional) | `utils/valuenorm.py` | port if trivial | Common MAPPO stabilizer |
| `PopArt` (optional) | `utils/popart.py` | port if trivial, else skip | Optional normalization |

---

## (c) OMITTED from port ❌

| Component / Extension | Why OMITTED | Tracked as |
|:--|:--|:--|
| `RIM` (Recurrent Independent Mechanisms) | Not in paper Fig.4 ; forked-repo addition | `D-marl-attention-extensions` (already declared) |
| `SCOFF` (Structured Coupled-Objective Factored RIMs) | Not in paper Fig.4 ; forked-repo addition | `D-marl-attention-extensions` |
| `SCOFFv2`, `RIMv2` | Variants of above | `D-marl-attention-extensions` |
| `set_transformer`, `attention_rim`, `sparse_attn`, etc. | 12 files in `utils/utilities/` — all upstream noise | `D-marl-attention-extensions` |
| `modularity.py` | SCOFF wrapper | `D-marl-attention-extensions` |
| `happo_trainer`, `hatrpo_trainer`, `mat_trainer` | Alternative algorithms not used by paper | `D-marl-algorithm-variants-omitted` (new) |
| `envs/mpe/` | Multi-agent Particle Env (not in paper) | `D-marl-alt-envs-omitted` (new) |
| `envs/starcraft2/`, `envs/hanabi/`, `envs/football/` | Alt envs (not in paper) | `D-marl-alt-envs-omitted` |
| `FocalLoss` class (`r_mappo.py:8-18`) | Defined but never called | `D-marl-focal-dead-code` (new) |
| `layer_output` Linear in `R_Actor_Meta` / `R_Critic_Meta` | Defined but never called | `D-marl-layer-output-unused` (E.4) |
| `RNNLayer_Meta.wager` + `sigmoid` (wager=True branch) | Dead code (grep confirmed) | `D-marl-rnn-meta-dead-wager-branch` (E.4) |
| `runner/shared/meltingpot_runner.py` | Syntax bug + wrong substrates | `D-marl-shared-runner-broken` (E.3) |
| `get_episode_parameters` + `probability=1.0` override | Dead code (while loop never runs) | `D-marl-episode-length` (E.3) |
| `min_max_normalize` substrate map | Wrong substrates (not paper's 4) | `D-marl-minmax-substrates-wrong` (E.3) |
| `render()` + GIF export | Development utility, not required for reproduction | — |
| `onpolicy/scripts/eval_*`, `render_*` | Out of paper-reproduction scope | — |
| `profile_run.prof`, `*.pyc`, `*.DS_Store` | Build artifacts | not tracked |
| All SC2 / Hanabi `.sh`, `.cc`, `.h` files | Not our envs | — |
| `dmlab2d-*.whl` | Binary wheel (will install dmlab2d separately via docs/install_linux.md) | already out of repo |
| Optimizers : AMS, AdamW, SWATS, SGD, RangerQH, Ranger, RAdam, Adagrad, Adadelta, RMSprop, RPROP | Paper T.12 specifies Adam only | only Adam + RangerVA kept |
| `use_popart` flag (all the `PopArt` integration) | Optional ; if port works without, skip | verify during E.8 |

---

## (d) Cascade scope — keep all 6 settings (FINAL decision)

**Paper §B.4 preamble** : *"MAPS not implemented fully, only with simple 2nd
order network with no cascade model due to limitations with computational
resources."*

**Student code** (`train_meltingpot.py:112-142`) : settings 2/4/5/6 activate
`cascade_iterations1` or `cascade_iterations2` = 50.

**Decision** : **port all 6 settings** with cascade support (iterations ≥ 1).

**Rationale :**
1. Same pattern as Blindsight/AGL where paper tables (T.9 hidden_dim=60 ;
   T.10 n_epochs_pre=60) disagreed with student code (40, 30) and **student
   code produced the paper numbers**. Paper prose summaries are lossy.
2. Cascade machinery is already implemented in `r_actor_critic.py` (via `for j
   in range(cascade_one):` loop around RNN forward) and in `r_actor_critic_meta.py`
   (same loop + cascade loop around SecondOrderNetwork). Dropping it means
   losing settings 2/4/5/6.
3. At `cascade_iterations = 1`, the loop runs once — no performance impact, no
   math difference vs no-cascade. So "no cascade" is just a special case of
   "cascade with 1 iteration".

**Config default** : `cascade_iterations1 = 1, cascade_iterations2 = 1`
(setting 1 baseline). Override per setting via CLI flag or YAML.

**Update to `deviations.md §B.11 D-marl-cascade-not-implemented`** :
- Previous verdict : `declared` (paper-admitted limitation).
- New verdict : ⚠️ paper prose inaccurate ; student code wins ; port all 6
  settings with cascade_iterations configurable (1 = no cascade, 50 = full).

---

## (e) Factorial settings mapping — 6 settings (confirmed E.3)

| Setting | `meta` | `cascade_iterations1` | `cascade_iterations2` | Label |
|:--:|:--:|:--:|:--:|:--|
| 1 | False | 1 | 1 | Baseline MAPPO |
| 2 | False | 50 | 1 | Cascade on 1st (no meta) |
| 3 | True | 1 | 1 | Meta only (no cascade) |
| 4 | True | 50 | 1 | **MAPS** (meta + cascade 1st) |
| 5 | True | 1 | 50 | Meta + cascade 2nd |
| 6 | True | 50 | 50 | Full (meta + cascade both) |

Port `Setting` dataclass (E.7) :
```python
@dataclass(frozen=True)
class MarlSetting:
    id: str          # '1', 'cascade_1st_only', 'maps', etc.
    label: str
    meta: bool
    cascade_iterations1: int   # 1 = disabled, 50 = full paper cascade
    cascade_iterations2: int
```

---

## (f) Target substrates — 4 substrates (confirmed E.3)

| Port id | MeltingPot substrate name | num_agents | Paper §A.4 label |
|:--|:--|:--:|:--|
| `commons_harvest_closed` | `commons_harvest__closed` | 6 | Commons Harvest Closed |
| `commons_harvest_partnership` | `commons_harvest__partnership` | 4 | Commons Harvest Partnership |
| `chemistry` | `chemistry__three_metabolic_cycles_with_plentiful_distractors` | 8 | Chemistry Three Metabolic Cycles with Plentiful Distractors |
| `territory_inside_out` | `territory_inside_out` | 5 | Territory Inside Out |

`config/env/marl/<port_id>.yaml` per substrate.

---

## (g) Paper-faithful adjustments applied during port (E.8+)

From `deviations.md §B.11` + E.2/E.3/E.4 audits. All target columns :

| Knob | Paper T.12 | Student default | Port target | Deviation ID |
|:--|:--:|:--:|:--:|:--|
| `hidden_size` | 100 | 144 | **100** | D-marl-hidden-size |
| `actor_lr` | 7e-5 | 7e-5 (config) / 2e-5 (shell) | **7e-5** | D-marl-actor-lr |
| `critic_lr` | paper typo "100" | 7e-5 | **7e-5** | D-marl-critic-lr |
| `entropy_coef` | 0.01 | 0.01 (config) / 0.004 (shell) | **0.01** | D-marl-entropy-coef |
| `num_env_steps` | 15e6 | 40e6 (config) / 300k (paper text) | **15e6** | D-marl-num-env-steps |
| `clip_param` | 0.2 | 0.2 | 0.2 | ✅ |
| `ppo_epoch` | 15 | 15 | 15 | ✅ |
| `weight_decay` | 1e-5 | 1e-5 | 1e-5 | ✅ |
| `optimizer` | Adam | Adam | **Adam** | ✅ |
| EMA `α` (wager) | 0.45 (eq.13) | 0.25 | **0.45** | D-marl-ema-alpha (E.3 new) |
| Wager target | `r_t > EMA_t` (eq.14) | `EMA > 0` | **paper eq.14** | D-marl-wager-condition (E.3 new) |
| `num_agents` | per substrate | per substrate | per substrate | ✅ |
| `cascade_iterations1/2` | 1 (paper prose) / 50 (student) | 1 or 50 (config) | **1 (default) or 50 (config override)** | D-marl-cascade-not-implemented (updated) |

---

## (h) Approx. port LOC estimate

| File | Student source | Port target | LOC estimate |
|:--|:--|:--|:--:|
| `src/maps/experiments/marl/__init__.py` | N/A | package init | 30 |
| `src/maps/experiments/marl/setting.py` | inlined | `MarlSetting` dataclass | 50 |
| `src/maps/experiments/marl/encoder.py` | `utils/cnn.py:1-185` | `CNNBase` + `CNNLayer` | 120 |
| `src/maps/experiments/marl/rnn.py` | `utils/rnn.py` + `utils/rnn_meta.py` | `RNNLayer`, `RNNLayerMeta` | 150 |
| `src/maps/experiments/marl/policy.py` | `r_actor_critic.py` + `r_actor_critic_meta.py` + `rMAPPOPolicy.py` | actor/critic + meta + MarlSecondOrderNetwork + wrapper | 500 |
| `src/maps/experiments/marl/trainer.py` | `r_mappo.py` + parts of runner | PPO trainer with EMA wager integration | 300 |
| `src/maps/experiments/marl/env.py` | `envs/meltingpot/MeltingPot_Env.py` | Env wrapper + `env_creator` | 250 |
| `src/maps/experiments/marl/runner.py` | `runner/separated/meltingpot_runner.py` | rollout + insert + wager EMA loop | 400 |
| `src/maps/experiments/marl/data.py` | inline | shared buffer types | 100 |
| `scripts/run_marl.py` | `scripts/train/train_meltingpot.py` | typer CLI | 200 |
| `scripts/slurm/marl_array.sh` | `train_meltingpot.sh` | DRAC GPU sbatch | 80 |
| `config/training/marl.yaml` + `config/env/marl/*.yaml` | config.py + shell | | 200 |

**Total port : ~2400 LOC.** Student source equivalent : ~2500-3000 LOC (after
dropping dead code + extensions). Roughly 1:1.

---

## (i) Order of execution (E.7+)

Sequential, per Rémy's discipline :

```
E.6  MeltingPot install DRAC (BLOCKER — before any port)
E.7  Scaffold src/maps/experiments/marl/ (empty modules + configs)
E.8  Port CNN + RNN + policy networks (no trainer yet)
E.9  Port trainer (PPO + EMA wager) + runner orchestration
E.10 Port MeltingPot env wrappers (deps installed via E.6)
E.11 Per-substrate configs + CLI integration
E.12 CLI + slurm templates end-to-end smoke
E.13 Parity tier-1 forward
E.14 Parity tier-2 buffer
E.15 Parity tier-3 update
E.16 DRAC smoke run (1 seed × 1 substrate × 100k steps)
E.17 Full run (24+ seeds × 4 substrates × 15M steps × 6 settings)
E.18 Aggregation + Table 7 compare + closeout
```

---

## (j) Frozen. E.5 DoD met.

Any scope changes after this lock require explicit re-approval from Rémy and
new audit sub-phase. No silent additions.

**Next : E.6 MeltingPot install verification on DRAC Tamia (critical blocker).**
