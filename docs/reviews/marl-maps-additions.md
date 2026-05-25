# Component review — MARL MAPS additions (paper eq.1-6 ↔ code)

**Phase E.4 — read-only audit.**
**Reviewer :** Rémy Ramadour + Claude, 2026-04-21.
**Files under review :**
- `external/paper_reference/marl_tmlr/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic_meta.py` (299 L) — already read in E.2.
- `external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/rnn_meta.py` (118 L) — used by `_Meta` variants.
- `external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/cnn.py` §CNNBase / CNNLayer — obs encoder.
- Paper §2.2 eq. 1–6, Fig. 4.

**DoD :** audit focused on paper↔code mapping of the MAPS-specific math. No code touched.

---

## (a) Paper equations 1–6

```
C_t  = X_t  − Ŷ_t^{(1)}              (eq.1)  comparator
C'_t = Dropout(C_t)                    (eq.2)  dropout on comparator
W_t  = W · C'_t + b                    (eq.3)  wagering — linear only, no activation
L_contrastive = ... SimCLR NT-Xent     (eq.4)  main-task loss
L_BCE = −[y·log σ(logits) + ...]       (eq.5)  wagering loss — per-unit BCE-with-logits
a_ir(t) = α·Σ w_ij·a_js(t) + (1-α)·a_ir(t-1)   (eq.6)  cascade update
```

Paper Fig. 4 : shows `ConvEncoder → Linear Layers → GRU → comparison matrix → wager units` + `→ Value Network`.

---

## (b) Cross-reference table

| Paper | Code location (student) | Port target (E.7–E.9) | Match |
|:--|:--|:--|:-:|
| **eq.1** `C = X − Ŷ` | `r_actor_critic_meta.py:190` `comparison_matrix = initial_states - actor_features_out` | `MAPSActor._compute_comparator(h_pre_rnn, h_post_rnn)` | ⚠️ semantic drift — see §(c) |
| **eq.2** `C' = Dropout(C)` | `r_actor_critic_meta.py:37` `comparison_out = self.dropout(torch.relu(self.comparison_layer(comparison_matrix)))` | `MarlSecondOrderNetwork.forward` | ⚠️ student adds `Linear + ReLU` before dropout (not in eq.2) |
| **eq.3** `W = W·C' + b` | `r_actor_critic_meta.py:41` `wager = self.wager(comparison_out)` | `MarlSecondOrderNetwork.wager_head` | ✅ linear only, no activation |
| **2 wager units** (Fig.4, eq.3) | `r_actor_critic_meta.py:24` `self.wager = nn.Linear(num_linear_units, 2)` | `wager_head = Linear(H, 2)` | ✅ |
| **eq.5** BCE-with-logits | `r_mappo.py:165, 221` `torch.nn.functional.binary_cross_entropy_with_logits(values_meta, wager_objective)` | `MAPPOTrainer._wager_loss` | ✅ |
| **eq.6** cascade on activations | `r_actor_critic_meta.py:38-39` (inside SecondOrderNetwork) + `rnn_meta.py:111-113` (inside RNNLayer_Meta) | per-network cascade hook | ✅ |
| **eq.4** contrastive / SimCLR | **NOT USED** in MARL — MAPPO uses policy loss (PPO clip) + value loss (MSE/Huber) ; no first-order reconstruction target here | SKIP | N/A |

---

## (c) 🚨 `comparator` semantics — MARL departs from paper eq.1

Paper eq.1 defines the comparator as `C_t = X_t − Ŷ_t^{(1)}` — input minus
*first-order network's output prediction*. For Blindsight/AGL this is
literally input-vs-reconstruction. For SARL it's input-vs-tied-weight-decoder.

MARL redefines it :

```python
# r_actor_critic_meta.py:183-190 inside R_Actor_Meta.evaluate_actions
actor_features = self.base(obs)                   # CNN encoder on obs
actor_features = self.layer_input(actor_features) # extra Linear
rnn_states     = self.layer_input(rnn_states)

initial_states = actor_features                   # snapshot PRE-RNN

actor_features_out = actor_features
rnn_states_out     = rnn_states
for j in range(self.cascade_one):
    actor_features_out, rnn_states_out, _ = self.rnn(actor_features_out, rnn_states_out, masks, ..., wager=False)

comparison_matrix = initial_states - actor_features_out   # PRE-RNN minus POST-RNN
```

**Observation** : MARL comparator = `pre_RNN_features − post_RNN_features`,
NOT `observation − reconstruction`. This is a **departure from paper eq.1**.

**Interpretation** : for MARL there is no "main-task reconstruction" signal —
the paper's §2.1 autoencoder-based comparator doesn't fit an RL policy. Student
approximates eq.1 by using the RNN's *delta* between input and output features
as a proxy for "how much does the memory change the representation" — which
maps onto the metacognitive intuition *"amount of update needed"*.

This is a **principled choice** even though it doesn't match eq.1 literally.
Port target : **preserve student behavior** (`comparator = h_pre_rnn − h_post_rnn`).
Flag as `D-marl-comparator-definition` new deviation.

---

## (d) 🚨 `SecondOrderNetwork` forward — student adds Linear + ReLU not in eq.2-3

Paper eq.2-3 reads : `C'_t = Dropout(C_t) ; W_t = W · C'_t + b`.

Student `SecondOrderNetwork.forward` :
```python
comparison_out = self.dropout(torch.relu(self.comparison_layer(comparison_matrix)))
```

That's `Dropout(ReLU(Linear(C)))` instead of `Dropout(C)`.

**Two extra ops over eq.2 :** a `Linear(H, H)` + ReLU applied **before** the
Dropout. The Linear layer has learnable weights `comparison_layer.weight`
initialized uniform(-1, 1).

This is **effectively a hidden layer** between the comparator and the wager
readout — analogous to the Pasquali 2010 hidden layer we had to restore for
Blindsight/AGL (D.25 RG-002). Here it's present by default — unlike
Blindsight/AGL where the same hidden layer had been dropped.

**Port target** : replicate exactly. Flag as `D-marl-comparison-layer-not-in-paper`
(not a blocker — MARL's hidden IS there ; paper prose just underspecifies).

---

## (e) Extra `layer_input` projection — not in paper

`R_Actor_Meta.__init__:66-67` + `R_Critic_Meta.__init__:219-220` define :
```python
self.layer_input  = nn.Linear(args.hidden_size, self.hidden_size)
self.layer_output = nn.Linear(self.hidden_size, args.hidden_size)
```

`layer_input` is used in `evaluate_actions` (L180-181) to project BOTH
`actor_features` and `rnn_states` before the RNN cascade + comparator snapshot.
`layer_output` is **never used** (dead code).

Shape is identity (`hidden_size → hidden_size`) so it's a learnable
input-conditioning projection — analogous to a "preprocessing" MLP. Not in
paper Fig.4 or §2.2.

**Port target** : keep `layer_input` for student parity (cheap 100×100
learnable identity-like projection), **drop `layer_output`** (dead).

---

## (f) `RNNLayer_Meta` — extra wager head is dead code

`rnn_meta.py:28,30` adds `self.wager = nn.Linear(H, 2)` + `self.sigmoid`
inside the RNN layer itself. The forward branches on `wager` flag :
```python
def forward(self, x, hxs, masks, prev_h1, prev_h2, cascade_rate1, cascade_rate2, wager=False):
    ...
    if wager:
        x = self.norm(x)
        if prev_h2 is not None:
            output_cascade2 = cascade_rate2*x + (1-cascade_rate2)*prev_h2
        else:
            output_cascade2 = x
        x = self.wager(output_cascade2)
        x = self.sigmoid(x)
        return x, hxs, output_cascade2
    else:
        ...
```

`grep -rn 'wager=True'` = 0 matches → **dead code**. All call-sites pass
`wager=False`.

**Port target** : drop the entire `wager=True` branch + `self.wager` +
`self.sigmoid` inside `RNNLayer_Meta`. Only keep the GRU + LayerNorm path.
Plus cascade update (`prev_h1` path) which IS active.

---

## (g) Where `wager_objective` enters the loss

Per E.3 audit (separated runner L197-217) :
```python
grad_rewards = α * rewards_values_tensor + (1 - α) * grad_rewards  # EMA
```
Then per-agent :
```python
wager_objective = 1 if grad_rewards.squeeze(1)[agent_id] > 0 else 0
```

Passed to `R_MAPPO.ppo_update` via `train()` call. Inside `ppo_update`
(`r_mappo.py:160-166`) :
```python
wager_objective = torch.tensor(wager_objective, ...).unsqueeze(-1).unsqueeze(0)
values_meta    = check(values_meta).to(**self.tpdv).squeeze(-1).squeeze(0).cuda()
wager_objective = check(wager_objective).to(**self.tpdv).squeeze(-1).squeeze(0).cuda()
loss_2 = torch.nn.functional.binary_cross_entropy_with_logits(values_meta, wager_objective)
loss_2_values = loss_2 * self.value_loss_coef
```

**Important details :**
1. **`values_meta`** comes from `R_Actor_Meta.evaluate_actions` which returns the
   wager directly (L196). So `values_meta = wager logits (raw, no sigmoid)`.
2. **`wager_objective` is binary {0, 1}** per agent per step, not one-hot
   `(1,0)/(0,1)` like paper eq.14. This is because `values_meta` has shape
   `(batch, 2)` but the wager_objective broadcasts scalar-per-agent → BCE
   per-unit still works via broadcasting.

Wait — actually `values_meta.squeeze(-1).squeeze(0)` flattens to `(batch,)` or
`(batch, 2)` squeezed ? Need to verify shapes. The comment at L164 says
`len(values_meta[0]), len(wager_objective[0])` — implies `values_meta[0]` is a
list-like of length ≥ something. Probably `(batch, 2)` after squeezes collapse
leading/trailing singletons.

**Action** : when porting (E.9), add explicit shape asserts at this loss call
to pin the expected dims. Student is sloppy on shapes.

---

## (h) Summary of deviations NEW from E.4

| ID | Location | Paper | Student | Verdict |
|:--|:--|:--|:--|:--|
| `D-marl-comparator-definition` | `r_actor_critic_meta.py:190` | eq.1 `C = X − Ŷ` | `C = h_pre_rnn − h_post_rnn` | ⚠️ principled approximation for RL (no reconstruction target) — preserve in port |
| `D-marl-comparison-layer-not-in-paper` | `SecondOrderNetwork.forward:37` | eq.2 `C' = Dropout(C)` | `C' = Dropout(ReLU(Linear(C)))` | ⚠️ extra hidden layer — KEEP (analogous to Blindsight D.25 restored Pasquali hidden) |
| `D-marl-layer-input-extra-projection` | `R_Actor_Meta.__init__:66` | not in Fig.4 | `layer_input = Linear(H, H)` pre-RNN | ⚠️ keep for student parity |
| `D-marl-rnn-meta-dead-wager-branch` | `rnn_meta.py:97-107` | N/A | `wager=True` branch never called | drop in port |
| `D-marl-layer-output-unused` | `R_Actor_Meta.__init__:67` | N/A | `layer_output = Linear(H, H)` never called | drop in port |

All these go into `deviations.md §B.11` with their respective verdicts during
E.18 closeout.

---

## (i) Port architectural summary (for E.8)

Assembled from E.2 + E.3 + E.4 :

```
MAPSActor(obs):
    h_cnn   = CNNBase(obs)                                 # conv encoder, hidden_size=100
    h_proj  = layer_input(h_cnn)                           # Linear(H, H) pre-RNN (keep)
    rnn_st  = layer_input(rnn_states)                      # Linear(H, H) on hidden state

    # Snapshot pre-RNN
    h_initial = h_proj                                     # (eq.1 X)

    h_out     = h_proj                                     # (RNN will update)
    rnn_out   = rnn_st
    for j in range(cascade_iterations1):                   # cascade on 1st (eq.6)
        h_out, rnn_out, _ = RNNLayer_Meta(h_out, rnn_out, masks, ..., wager=False)

    actions, logprobs = ACTLayer(h_out)                    # normal policy head

    # Wager path (only during eval_actions for training) :
    comparator = h_initial − h_out                         # (eq.1 — MARL variant)
    for j in range(cascade_iterations2):                    # cascade on 2nd (eq.6)
        wager, prev_comp = MarlSecondOrderNetwork(comparator, prev_comp, cascade_rate2)
    # wager is raw logits (B, 2), goes to BCE-with-logits vs wager_objective

MarlSecondOrderNetwork(comparator, prev_comp, cascade_rate):
    h = Dropout(ReLU(Linear_compare(comparator)))          # (eq.2 + extra hidden layer)
    if prev_comp is not None:
        h = cascade_rate * h + (1 - cascade_rate) * prev_comp   # (eq.6)
    wager_logits = Linear_wager(h)                         # (eq.3), 2 units raw
    return wager_logits, h
```

```
MAPSCritic(cent_obs):
    h_cnn  = CNNBase(cent_obs)                             # WORLD.RGB encoded
    h_proj = layer_input(h_cnn)
    for j in range(cascade_iterations1):
        h_proj, rnn_out, _ = RNNLayer_Meta(h_proj, rnn_out, masks, ..., wager=False)
    value = v_out(h_proj)
    return value, rnn_out
```

No 2nd-order on critic side (paper Fig.4 only puts wager on actor side).

---

## (j) E.4 open questions closed

- **Q1 (from E.2)** : `layer_input` — NOT in paper Fig.4, keep for student parity (this doc §e).
- **Q2 (from E.2)** : `layer_output` — dead code, drop.
- **Q3 (from E.2)** : `FocalLoss` in r_mappo.py — dead code (E.2 already noted), drop.
- **Q4 (from E.2)** : runner branching — **resolved in E.3** (separated only, settings 1-6 switch `meta`).
- **Q5 (from E.2)** : `wager_objective` EMA formula — **resolved in E.3** (α=0.25 student, paper eq.13 says α=0.45 ; port target = paper).

---

## (k) Effort estimate for E.8 + E.9 ports (informed by E.2-E.4)

| Task | Eng effort | Notes |
|:--|:--:|:--|
| `CNNBase` + `CNNLayer` port | ~2-3 h | 758 L student file has other encoder variants we skip |
| `RNNLayer` (baseline) + `RNNLayer_Meta` (drop wager branch) | ~2 h | GRU + LayerNorm + cascade update |
| `MarlSecondOrderNetwork` (with comparison_layer) | ~1 h | Port of student's 25-line class |
| `MAPPOActor` / `MAPSActor` / `MAPPOCritic` / `MAPSCritic` | ~4-6 h | Wire CNN + RNN + cascade + (wager for meta) |
| `MAPPOPolicy` wrapper (actor + critic + actor_meta + critic_meta + 7 optim variants → 2 chosen) | ~2 h | Adam + RangerVA only |
| `MAPPOTrainer` (PPO clip + value loss + meta BCE) | ~4-6 h | Plus proper shape asserts |

**Total E.8 + E.9 : 15-20 eng hours.** Matches plan estimate.

---

**E.4 DoD met. 5 new deviations documented. No code touched. Next : E.5 decision
lock on RIM/SCOFF/skill-dynamics extensions (~1 h) then E.6 MeltingPot DRAC
install (critical blocker).**
