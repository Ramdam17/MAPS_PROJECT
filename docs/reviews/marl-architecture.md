# Component review — MARL / MAPPO core architecture

**Phase E.2 — read-only audit.**
**Reviewer :** Rémy Ramadour + Claude, 2026-04-21.
**Files under review :**
- `external/paper_reference/marl_tmlr/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py` (294 L) — standard actor + critic.
- `external/paper_reference/marl_tmlr/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic_meta.py` (299 L) — MAPS-extended actor + critic + 2nd-order.
- `external/paper_reference/marl_tmlr/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py` (467 L) — policy wrapper, manages 4 networks.
- `external/paper_reference/marl_tmlr/onpolicy/algorithms/r_mappo/r_mappo.py` (316 L) — PPO trainer with MAPS wager integration.
- **Total : 1376 L.**

**Paper sources :** §2.2 (MAPS architecture), Fig. 4 (MARL diagram), §B.4 + Table 12 (hyperparams), Table 7 (z-scores).

**DoD :** audit document produced, 0 code touched.

---

## (a) High-level architecture

**Four networks maintained in parallel by `R_MAPPOPolicy`** :

```
                ┌─────────────────────┐     ┌─────────────────────┐
                │ R_Actor (baseline)  │     │ R_Actor_Meta (MAPS) │
                │  obs → CNN → RNN →  │     │  obs → CNN → RNN_Meta → ACT
                │  ACT (action head)  │     │                     │
                │                     │     │  + SecondOrderNetwork
                │                     │     │    (comparator + wager)
                └─────────────────────┘     └─────────────────────┘

                ┌─────────────────────┐     ┌─────────────────────┐
                │ R_Critic (baseline) │     │ R_Critic_Meta (MAPS)│
                │  obs → CNN → RNN →  │     │  obs → CNN → RNN_Meta →
                │  v_out              │     │  v_out              │
                │                     │     │                     │
                │                     │     │  (no 2nd-order on   │
                │                     │     │   critic side — paper Fig.4)
                └─────────────────────┘     └─────────────────────┘
```

Each setting picks **one of the two parallel branches** at rollout time :
- `setting.meta=False` → uses `actor` + `critic` (baseline MAPPO).
- `setting.meta=True` → uses `actor_meta` + `critic_meta` (MAPS).

**Implication for port :** both branches coexist in the policy wrapper, and both
get trained simultaneously during `ppo_update` (with `meta=True`). This is ≈ 2×
parameter count compared to running only one branch.

Per `rMAPPOPolicy.py:44-47` :
```python
total_parameters_normal = count_parameters(self.actor) + count_parameters(self.critic)
total_parameters_meta   = count_parameters(self.actor_meta) + count_parameters(self.critic_meta)
print("parameters normal ", total_parameters_normal, " |||  parameters meta ", total_parameters_meta)
```

---

## (b) Module-by-module map for port

| Student module | Student file:line | Paper ref | Port target (Phase E.7+) |
|:--|:--|:--|:--|
| `R_Actor` | `r_actor_critic.py:15-179` | Fig.4 (actor box) | `src/maps/experiments/marl/policy.MAPPOActor` |
| `R_Critic` | `r_actor_critic.py:183-294` | Fig.4 (critic box) | `src/maps/experiments/marl/policy.MAPPOCritic` |
| `R_Actor_Meta` | `r_actor_critic_meta.py:45-196` | Fig.4 + §2.2 | `src/maps/experiments/marl/policy.MAPSActor` |
| `R_Critic_Meta` | `r_actor_critic_meta.py:199-299` | Fig.4 + §2.2 | `src/maps/experiments/marl/policy.MAPSCritic` |
| `SecondOrderNetwork` (MARL) | `r_actor_critic_meta.py:17-42` | §2.2 eq.1-3, Fig.4 | `src/maps/experiments/marl/policy.MarlSecondOrderNetwork` (NEW) |
| `R_MAPPOPolicy` | `rMAPPOPolicy.py:12-468` | N/A (glue) | `src/maps/experiments/marl/policy.MAPPOPolicy` |
| `R_MAPPO` (trainer) | `r_mappo.py:20-317` | §2.2 eq.13-14 + PPO | `src/maps/experiments/marl/trainer.MAPPOTrainer` |

**Note :** the MARL `SecondOrderNetwork` is **structurally different** from
Blindsight/AGL/SARL — it has a real `comparison_layer` (L22) with `Linear +
ReLU + Dropout`, whereas the other domains use element-wise subtraction or
tied-weight decoder. Port cannot reuse `maps.components.SecondOrderNetwork`
as-is ; needs a new class. Tracked in `deviations.md §B.11` as an architectural
note.

---

## (c) Cascade in MARL

Student code **contains the cascade machinery** but, per paper §B.4 preamble :

> *"MAPS not implemented fully, only with simple 2nd order network with no
> cascade model due to limitations with computational resources."*

This means **`cascade_iterations1 = cascade_iterations2 = 1`** in all production
runs → `cascade_rate = 1.0 / 1 = 1.0` → the loop `for j in range(1)` runs once
→ degenerate "no-cascade" path.

Port design decision (E.5) : **OMIT cascade machinery entirely** from the port.
Don't carry dead code. Single forward pass through the RNN layer suffices.
Tracked as `D-marl-cascade-not-implemented` in deviations.md (already declared).

---

## (d) Attention extensions (RIM / SCOFF) — OMIT in port

`R_Actor.__init__` L59-72 gates between 3 RNN backends :

```python
if self.use_attention and len(self._obs_shape) >= 3:
    if self._attention_module == "RIM":
        self.rnn = RIM(...)
    elif self._attention_module == "SCOFF":
        self.rnn = SCOFF(...)
elif not self.use_attention:
    self.rnn = RNNLayer(...)  # plain GRU, paper Fig.4
```

Paper Fig.4 depicts only the plain-GRU path. Student's forked repo adds RIM,
SCOFF, skill_dynamics, bottom-up attention — these **are not part of MAPS
architecture**, they come from the MAPPO-ATTENTION upstream repo.

**Port decision (E.5)** : use `RNNLayer` only (plain GRU + sinusoidal pos enc).
Skip `RIM` / `SCOFF` / `modularity.py` / `rim_cell.py` entirely. Tracked as
`D-marl-attention-extensions` (already declared as "policy OMIT" in deviations).

---

## (e) `SecondOrderNetwork` (MARL variant) — detail

```python
class SecondOrderNetwork(nn.Module):
    def __init__(self, num_linear_units):
        super().__init__()
        self.comparison_layer = nn.Linear(num_linear_units, num_linear_units)
        self.wager = nn.Linear(num_linear_units, 2)          # ← 2 units (paper eq.3)
        self.dropout = nn.Dropout(p=0.1)                     # ← 10% (not 50% like Blindsight)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        torch.nn.init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):
        comparison_out = self.dropout(torch.relu(self.comparison_layer(comparison_matrix)))
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
        wager = self.wager(comparison_out)
        return wager, comparison_out
```

**Key properties** :
- **2-unit wager output** — matches paper eq.3 (raw logits, no activation).
  Downstream loss uses `binary_cross_entropy_with_logits` (r_mappo.py L165, L221).
- **`comparison_layer` = Linear + ReLU + Dropout(0.1)** — different from
  Blindsight/AGL (which use element-wise subtraction directly) and from
  SARL (tied-weight decoder).
- **Comparator input** (per R_Actor_Meta L186-194) = `initial_states - actor_features_out` :
  difference between pre-RNN and post-RNN actor features. Not input-vs-reconstruction.
  → This is the **MARL-specific "comparator" definition**.
- Input dim = `hidden_size` (paper T.12 : 100 ; student config : 144).

---

## (f) `R_Actor_Meta.evaluate_actions` — how the wager is produced

```python
def evaluate_actions(self, obs, rnn_states, action, masks, ...):
    ...
    actor_features = self.base(obs)                          # CNN encoder
    actor_features = self.layer_input(actor_features)        # extra Linear !
    rnn_states = self.layer_input(rnn_states)

    initial_states = actor_features                          # snapshot pre-RNN

    for j in range(self.cascade_one):                        # L186 — 1 iter in prod
        actor_features_out, rnn_states_out, _ = self.rnn(
            actor_features_out, rnn_states_out, masks, ...
        )

    comparison_matrix = initial_states - actor_features_out  # pre vs post-RNN

    prev_comparison = None
    for j in range(self.cascade_two):                        # L193 — 1 iter in prod
        wager, prev_comparison = self.second_order(
            comparison_matrix, prev_comparison, self.cascade_rate_two
        )

    return wager
```

**Observations** :
- `self.layer_input` (L66) : **extra Linear layer** before RNN, not mentioned
  in paper. Identity-like (`hidden_size → hidden_size`), serves as a learnable
  projection. Present in both `_Meta` variants.
- The method is named `evaluate_actions` but **returns `wager` only** (no action
  logprobs / entropy). Misleading naming — used by trainer as a wager getter.

---

## (g) `ppo_update` wager integration

```python
# r_mappo.py L149-156
values_meta = self.policy.evaluate_actions_meta(share_obs_batch, obs_batch,
    rnn_states_batch, rnn_states_critic_batch, actions_batch, masks_batch, ...)

# L160-166
wager_objective = torch.tensor(wager_objective, ...).unsqueeze(-1).unsqueeze(0)
loss_2 = torch.nn.functional.binary_cross_entropy_with_logits(
    values_meta, wager_objective
)
loss_2_values = loss_2 * self.value_loss_coef  # ← scaled by value_loss_coef (paper T.12 = 1.0)

# L196-199 : if meta=True, backprop through actor_meta
if meta:
    loss_2_values.backward(retain_graph=True)
    self.policy.actor_meta_optimizer.step()

# Then standard PPO actor update
total_loss = (policy_loss - dist_entropy * self.entropy_coef)
total_loss.backward()
```

**Same flow for critic** (L211-233) : `values_meta_critic = evaluate_actions_meta(...)`,
compute `loss_2_critic`, if `meta`: backprop + step critic_meta_optimizer.

**Wager objective provenance** : `wager_objective` is passed as an argument to
`train()` from the runner (see `onpolicy/runner/separated/meltingpot_runner.py`).
It is the per-step EMA-based signal from paper eq.13-14 (high wager if reward >
EMA, low otherwise) — computed in the runner outside the trainer.

---

## (h) Hyperparameters — paper T.12 vs student config

| Knob | Paper T.12 | Student `onpolicy/config.py` default | Port target |
|:--|:--:|:--:|:--:|
| `hidden_size` | 100 | 144 | **100** (paper) |
| `actor_lr` | 7e-5 | 7e-5 (config) BUT shell passes `--lr 2e-5` | **7e-5** (paper) |
| `critic_lr` | "100" (paper typo) | 7e-5 (= actor_lr) | **7e-5** (student + paper-consistent reading) |
| `entropy_coef` | 0.01 | 0.01 (config) BUT shell passes `--entropy_coef 0.004` | **0.01** (paper) |
| `clip_param` | 0.2 | 0.2 | 0.2 |
| `ppo_epoch` | 15 | 15 | 15 |
| `weight_decay` | 1e-5 | 1e-5 | 1e-5 |
| `num_env_steps` | 15e6 | 40e6 (config) / 300k (paper text p.15) | **15e6** (paper T.12, decision point) |
| `optimizer` | Adam | Adam (from shell) | Adam |
| `opti_eps` | paper silent | 1e-5 | 1e-5 |
| `max_grad_norm` | paper silent | 10.0 | 10.0 |
| `data_chunk_length` | paper silent | 10 | 10 |
| `value_loss_coef` | paper silent | 1.0 | 1.0 |
| Cascade iterations | 1 (paper preamble "no cascade") | 1 (cascade_iterations1/2 = 1) | **1** → omit the machinery |
| Attention module | plain GRU (Fig.4) | `use_attention=False` for paper | **RNNLayer only** |

**Known deviations** (already in `deviations.md §B.11`) — all flagged 🆘
(student config drifts from paper text, student shell wrappers drift from
student config) :
- `D-marl-hidden-size` : student 144 vs paper 100.
- `D-marl-actor-lr` / `D-marl-critic-lr` / `D-marl-entropy-coef` : shell vs
  config vs paper mismatches.
- `D-marl-num-env-steps` : paper T.12 says 15e6 but paper text p.15 says 300k
  — our port target = 15e6 (Table authoritative per B.11 audit note).

---

## (i) Items to port (scope for E.7-E.12)

| Phase | Scope | Deliverable |
|:--|:--|:--|
| E.7 | Package skeleton | `src/maps/experiments/marl/{__init__,setting,data}.py`, config YAMLs |
| E.8 | `MAPPOActor`, `MAPPOCritic`, `MAPSActor`, `MAPSCritic`, `MarlSecondOrderNetwork` | `policy.py` (~300-400 L) |
| E.9 | `MAPPOTrainer` with PPO clip + meta wager integration | `trainer.py` (~300 L) |
| E.10 | MeltingPot env wrapper | `env.py` |
| E.11 | Per-substrate reward/termination configs | `config/env/marl/*.yaml` |
| E.12 | CLI + slurm | `scripts/run_marl.py`, `scripts/slurm/marl_array.sh` |

**Items to SKIP per E.5 decision** :
- `RIM` / `SCOFF` / `modularity.py` attention extensions.
- Cascade machinery (dead code at cascade_iter=1).
- `HAPPO`, `HATRPO`, `MAT` algorithm variants (not part of paper MAPS protocol).
- `PopArt` (optional, not critical ; can add later if needed).
- `skill_dynamics`, `transition_model`, `bottom_up` attention.

---

## (j) Open questions surfaced by audit

1. **`layer_input` extra Linear** in `R_Actor_Meta` / `R_Critic_Meta` (L66-67,
   L219-220) — not mentioned in paper Fig.4. Identity-initialized learnable
   projection. Port : **include** for student-parity ; could ablate later.
2. **Two parallel networks at once** (actor + actor_meta + critic + critic_meta)
   — confirm that runner actually uses only one branch per setting, or uses both
   simultaneously. To clarify in E.3 (runner audit).
3. **Per-setting branching logic** : where does the code choose `meta=True/False`
   ? To clarify in E.3 (runner) and E.8 (port decision whether to have 2 separate
   classes or a single parametric class).
4. **`wager_objective` computation** : EMA signal produced in runner. Exact
   formula and smoothing factor (`α=0.45` per paper eq.13) to be audited in E.3.
5. **`FocalLoss` class** at top of `r_mappo.py` (L8-18) — defined but **never
   used** in the trainer body. Dead code — **SKIP in port**.

---

## (k) Deliverables from E.2 (this commit)

- ✅ `docs/reviews/marl-architecture.md` (this doc).
- ✅ Module map, cascade + attention decisions documented.
- ✅ Open questions flagged for E.3 / E.4 / E.5.
- ✅ Hyperparameter target table aligned with paper T.12.

**DoD met. No code touched. Next : E.3 audit MeltingPot env wiring + runner
orchestration.**
