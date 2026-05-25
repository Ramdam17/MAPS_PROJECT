# Component review — MARL env wiring + runner orchestration

**Phase E.3 — read-only audit.**
**Reviewer :** Rémy Ramadour + Claude, 2026-04-21.
**Files under review :**
- `external/paper_reference/marl_tmlr/onpolicy/envs/meltingpot/MeltingPot_Env.py` (360 L).
- `external/paper_reference/marl_tmlr/onpolicy/scripts/train/train_meltingpot.py` (260 L) — entry + settings 1-6.
- `external/paper_reference/marl_tmlr/onpolicy/runner/shared/meltingpot_runner.py` (341 L) — broken/unused.
- `external/paper_reference/marl_tmlr/onpolicy/runner/separated/meltingpot_runner.py` (808 L) — **production runner** with meta logic.
- `external/paper_reference/marl_tmlr/train_meltingpot.sh` (312 L) — shell wrapper.
- **Total : 2081 L.**

**Paper sources :** §A.4 (env descriptions, 4 substrates), §2.2 eq.13-14 (EMA wagering), Table 8 (env properties), Figs.11.

**DoD :** audit document produced, 0 code touched.

---

## (a) Environment wrapper : `MeltingPotEnv`

**File :** `MeltingPot_Env.py:124-299`.

Wraps a **dmlab2d** substrate as a **rllib MultiAgentEnv** with dict-per-agent
observation / action / reward / done. Key methods :

| Method | Purpose | Notes |
|:--|:--|:--|
| `__init__(env, max_cycles=400)` | Build from a pre-constructed dmlab2d substrate. Infer `num_players`, observation_space (per-agent RGB), action_space, share_observation_space (WORLD.RGB per-agent). | MAX_CYCLES = 400 (hardcoded). |
| `reset(*args, **kwargs)` | Return `(obs_dict, {})`. obs_dict : `{'player_{i}': {'RGB':…, 'WORLD.RGB':…}}`. | |
| `step(action_dict, rewards_dict=None)` | Step all agents ; return `(obs, rewards_dict, dones_dict, info)`. Handles multi-rollout loop via `actions.shape[1]` inner loop (one env step per rollout thread). Truncation at `num_cycles >= max_cycles`. | Complex : stacks observations along rollout dim. |
| `close()` / `render()` / `_convert_spaces_tuple_to_dict` | rllib boilerplate. | |

**Observation spec** :
- Per-agent ego RGB : **`(11, 11, 3)` uint8** after downsampling (from raw 88×88 with scale factor 8).
- Centralized WORLD.RGB : varies per-substrate, e.g. `(168, 168, 3)` for `territory__rooms`.
- `_OBSERVATION_PREFIX = ['WORLD.RGB', 'RGB']` — exactly 2 keys kept per-agent.

**Downsampling** : `DownSamplingSubstrateWrapper` (L326-350) wraps the substrate to
reduce RGB from 88×88 to 11×11 via `cv2.resize(..., INTER_AREA)`. Scale factor
configurable via `env_config['scaled']` (paper default = 8).

**Env creator** L352-360 :
```python
def env_creator(env_config):
    env = meltingpot_substrate.build(env_config['substrate'], roles=env_config['roles'])
    env = DownSamplingSubstrateWrapper(env, env_config['scaled'])
    env = MeltingPotEnv(env)
    return env
```

---

## (b) 4 target substrates (paper §A.4)

| Paper name | Code `substrate_name` | Num agents | Type |
|:--|:--|:--:|:--|
| Commons Harvest Closed | `commons_harvest__closed` | 6 | Social dilemma + enclosed |
| Commons Harvest Partnership | `commons_harvest__partnership` | 4 | 1:many cooperation |
| Chemistry Three Metabolic Cycles with Plentiful Distractors | `chemistry__three_metabolic_cycles_with_plentiful_distractors` | 8 | Equilibrium selection |
| Territory Inside Out | `territory_inside_out` | 5 | Bargaining + zap |

From `get_episode_parameters` (separated runner L53-90) :
- `commons_harvest__closed`     : probability=0.15, episode_length=1000
- `commons_harvest__partnership` : probability=0.15, episode_length=1000
- `chemistry__three_metabolic_cycles_with_plentiful_distractors` : probability=0.2, episode_length=1000
- `territory_inside_out`        : probability=0.2, episode_length=1000

**Not yet clarified** : exact `default_player_roles` per substrate. Student
relies on `meltingpot.substrate.get_config(name).default_player_roles` — the
MeltingPot package provides this at install time.

---

## (c) Settings 1-6 (student train_meltingpot.py L112-142)

| Setting | `meta` | `cascade_iterations1` | `cascade_iterations2` | Paper mapping |
|:--:|:--:|:--:|:--:|:--|
| 1 | False | 1 | 1 | Baseline (no meta, no cascade) |
| 2 | False | 50 | 1 | Cascade on 1st (no meta) |
| 3 | True | 1 | 1 | Meta only (no cascade) |
| 4 | True | 50 | 1 | **MAPS** (meta + cascade 1st) |
| 5 | True | 1 | 50 | Meta + cascade 2nd |
| 6 | True | 50 | 50 | Full (meta + cascade both) |

**🚨 Paper-vs-code contradiction** : Paper §B.4 preamble says *"MAPS not
implemented fully, only with simple 2nd order network with **no cascade** model
due to limitations with computational resources."* But student code at
`train_meltingpot.py:114-142` lets cascade_iterations go up to 50 for settings
2/4/5/6. **Port must decide** whether to :
- (a) Respect paper preamble → port only settings 1 + 3 (no cascade anywhere).
- (b) Respect student code → port all 6 settings with cascade = 50 iters when enabled.

**Recommendation (E.5 decision review)** : Go with student code (all 6 settings),
matching the Blindsight/AGL pattern (*"student code > paper tables/prose for
reproduction"*). Cascade machinery ALREADY present in `r_actor_critic.py` and
reads like a functional loop. Flag as open decision pending ablation evidence.
**Update D-marl-cascade-not-implemented in deviations.md accordingly.**

---

## (d) `wager_objective` computation (paper eq.13-14)

**Student `runner/separated/meltingpot_runner.py:191-217`** (inside per-step loop) :

```python
if self.all_args.meta:
    if step < episode_length - 1:
        rewards_values = np.array(list(rewards[0].values()))
        alpha = 0.25                              # ← EMA hyperparameter
        rewards_values_tensor = []

        for agent_id in range(self.num_agents):
            rewards_values_tensor.append((self.buffer[agent_id].rewards)[step])
            list_rewards[agent_id].append((self.buffer[agent_id].rewards)[step])

        rewards_values_tensor = torch.from_numpy(np.array(rewards_values_tensor)).squeeze(1)

        # EMA update
        grad_rewards = alpha * rewards_values_tensor + (1 - alpha) * grad_rewards
        grad_list.append(grad_rewards)
```

Then in `meta()` L338 :
```python
wager_objective = 1 if grad_rewards.squeeze(1)[agent_id] > 0 else 0
```

### 🚨 Paper eq.13-14 vs student code — 2 deviations

| Knob | Paper eq.13-14 | Student code | Deviation ID |
|:--|:--|:--|:--|
| EMA α | **0.45** (paper §2.2) | **0.25** (student L197) | new `D-marl-ema-alpha` |
| Wager condition | **r_t > EMA_t** (eq.14: `(1,0) if r_t > EMA_t`) | **EMA_t > 0** (L338 `grad_rewards > 0`) | new `D-marl-wager-condition` |

**Interpretation** : Student's `grad_rewards > 0` checks *"is the running average reward
positive?"*, NOT *"is the current reward above the running average?"*. Those are
functionally different signals :
- Paper (eq.14) : high wager when agent is above its own baseline — a
  *self-comparison* signal that tracks short-term performance above the historical mean.
- Student : high wager when the running mean is positive — a global "things
  are going OK" signal.

**Port target** : implement paper eq.13-14 literally (α=0.45, y = (1,0) if r_t
> EMA_t, (0,1) else), log student behavior as available-via-config for ablation.

---

## (e) Architectural note : `setting == 2` has cascade but no meta

Settings 2 (cascade only, no meta) uses the **baseline `R_Actor` / `R_Critic`**
(with cascade loop running inside `self.rnn`). So cascade IS applied even
without meta — it's a cascade ON the RNN states, not only in the SecondOrderNetwork.

This is **different from Blindsight/AGL** where cascade only lives in the
second-order forward pass (cascade_iterations in `SecondOrderNetwork.forward`
iterating comparator updates). MARL puts cascade INSIDE the RNN layer's
forward for the actor/critic features.

Implication : port must implement cascade both :
1. **Inside RNN** (for settings 2, 4, 6) — on actor_features / critic_features.
2. **Inside SecondOrderNetwork** (for settings 5, 6) — on comparison_matrix.

Paper §2.2 eq.6 uses cascade update uniformly, not distinguishing between 1st
and 2nd order placement. MARL student code places it in both, parametrized by
`cascade_iterations1` and `cascade_iterations2`. Port must follow.

---

## (f) Shared vs separated runner

- **`shared/meltingpot_runner.py`** (341 L) : uses `self.trainer.policy.get_actions`
  (baseline only), **has a syntax bug** (L61 `random.ramdom()` typo), no meta
  logic. Designed for `--share_policy` but broken.
- **`separated/meltingpot_runner.py`** (808 L) : implements the full MAPS
  protocol with EMA wagering. This is the **production runner** used by the paper.

Student `train_meltingpot.py:230-233` picks :
```python
if all_args.share_policy:
    from onpolicy.runner.shared.meltingpot_runner import MeltingpotRunner
else:
    from onpolicy.runner.separated.meltingpot_runner import MeltingpotRunner
```

Paper shell wrapper (`train_meltingpot.sh`) passes `--share_policy` as `False`
or omits it (default False) → always uses separated runner.

**Port decision** : implement only the separated path. Mark `--share-policy` as
deprecated / NotImplemented in CLI for now.

---

## (g) Episode length / probability

Student `runner/separated/meltingpot_runner.py:136-141` :
```python
probability, episode_length = get_episode_parameters(self.substrate_name)
probability = 1.0                                # ← OVERRIDE ! ignores map
while random.random() > probability and episode_length < self.episode_length:
    episode_length += 100
print("episode #{} length: {}".format(episode, episode_length))
```

`probability = 1.0` OVERRIDES the map (`commons_harvest__closed: 0.15`) → the
while loop **never runs** (`random.random()` ∈ [0,1) always ≤ 1.0 → `> 1.0` is
always False). So `episode_length = 1000` constant.

**Dead code cleanup for port** : drop `get_episode_parameters` helper, use a
flat `episode_length = 1000` constant (or `cfg.training.episode_length`).

---

## (h) Rewards handling

Rewards come from `MeltingPotEnv.step` as a dict `{player_0: float, ..., player_N: float}`.

Student runner L194 :
```python
rewards = np.array([player_dict[f'player_{i}'] for player_dict in rewards for i in range(self.num_agents)], dtype=np.float32)
```

Transposed to per-agent per-rollout-thread array, fed into buffer.

Shared runner min-max normalization map (L332-342) has **wrong substrates** :
```python
minmax_map = {
    'allelopathic_harvest__open': {'min': -17.8, 'max': 92.4},
    'clean_up': {'min': 0.0, 'max': 188.6},
    'prisoners_dilemma_in_the_matrix__arena': {'min': 0.9, 'max': 22.8},
    'territory__rooms': {'min': 10.4, 'max': 236.3}
}
```

None of these are the 4 paper substrates. Paper uses `commons_harvest__closed`,
`commons_harvest__partnership`, `chemistry__three_metabolic_cycles_with_plentiful_distractors`,
`territory_inside_out`. **Min-max normalization logic is UNUSED for paper runs.**

**Port decision** : skip `performance_score` / `min_max_normalize` — compute
raw episode rewards and z-scores vs paper Table 7 directly.

---

## (i) Hyperparameters actually used (shell wrapper vs config)

Per `train_meltingpot.sh` (partially read, 312 L) — shell passes per-run args
that **override** config defaults. Audit decision (B.11) flags :

| Arg | Shell wrapper | `onpolicy/config.py` default | Paper T.12 | Chosen default for port |
|:--|:--:|:--:|:--:|:--:|
| `--lr` (actor_lr) | `0.00002` = 2e-5 | 7e-5 | **7e-5** | **7e-5** (paper) |
| `--entropy_coef` | 0.004 | 0.01 | **0.01** | **0.01** (paper) |
| `--num_env_steps` | variable | 40e6 | **15e6** | **15e6** (paper T.12) |
| `--hidden_size` | uses shell | 144 | **100** | **100** (paper) |
| `--ppo_epoch` | 15 | 15 | 15 | 15 |
| `--clip_param` | 0.2 | 0.2 | 0.2 | 0.2 |

Already tracked as D-marl-* in deviations.md. Phase E.8-E.12 port aligns all to paper T.12.

---

## (j) Items surfaced for follow-up sub-phases

| Item | Phase | Priority |
|:--|:--|:--:|
| EMA α = 0.45 (paper) vs 0.25 (student) — new D-marl-ema-alpha | E.9 | 🆘 |
| Wager condition `r_t > EMA_t` (paper) vs `EMA_t > 0` (student) — new D-marl-wager-condition | E.9 | 🆘 |
| Cascade placement : inside RNN (for 1st) + inside SecondOrderNetwork (for 2nd) | E.8-E.9 | — |
| Port only separated runner path (shared is broken) | E.9 | — |
| Skip `get_episode_parameters` + min-max normalize (dead code) | E.9 | — |
| Substrates covered : 4 paper + unused mini-list (audit only, no port of extras) | E.10-E.11 | — |
| `FocalLoss` in r_mappo.py : dead code (never called) | E.9 | — |

---

## (k) Updated deviations (to add in E.12 commit)

**New deviations surfaced in E.3** :

| ID | Location | Paper | Student | Port target | Verdict |
|:--|:--|:--|:--|:--|:--|
| D-marl-ema-alpha | `runner/separated/meltingpot_runner.py:197` | **α=0.45** (eq.13) | α=0.25 | 0.45 | 🆘 |
| D-marl-wager-condition | `runner/separated/meltingpot_runner.py:338` | **r_t > EMA_t** (eq.14) | EMA_t > 0 | eq.14 literal | 🆘 |
| D-marl-episode-length | `runner/separated/meltingpot_runner.py:136-141` | paper silent | 1000 (via dead code) | 1000 | ⚠️ |
| D-marl-shared-runner-broken | `runner/shared/meltingpot_runner.py:61` | N/A | syntax bug `random.ramdom()` | port separated only | ⚠️ |
| D-marl-minmax-substrates-wrong | `runner/shared/meltingpot_runner.py:336` | 4 paper substrates | 4 different substrates | compute raw z-scores vs T.7 | ⚠️ |

These go into `deviations.md §B.11` closing pass during E.18.

---

## (l) Next phases dependencies resolved

- **E.4 (MAPS additions audit)** — already covered partially by E.2 (SecondOrderNetwork class)
  + this doc (wager_objective EMA computation). Short focused review expected.
- **E.5 (RIM/SCOFF decision)** — already locked in E.2. Plus : new decision lock on
  cascade (port all 6 settings w/ cascade up to 50 iters, despite paper preamble).
- **E.6 (MeltingPot install)** — independent of audit scope ; next critical blocker.

**E.3 DoD met. No code touched. Next : E.4 (MAPS-additions-specific audit, short).**
