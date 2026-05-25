# Reverse-prompt — `src/maps/domains/marl/policy.py`

**Module path actuel (sur main) :** `src/maps/experiments/marl/policy.py`
**Taille :** ~603 lignes.
**Paper :** §2.2 eq.1-3 + Table 12 + Fig.4.

---

## 1. Quelle aurait été la spec ?

> Écris le module `policy.py` avec **6 classes** :
>
> **1. `MarlSecondOrderNetwork`** (paper §2.2 eq.1-3 + MARL extra
> `comparison_layer`) :
> - `comparison_layer = Linear(hidden, hidden)` + ReLU + Dropout(0.1)
> - Cascade update (paper eq.6)
> - `wager_layer = Linear(hidden, 2)` raw logits
> - Init `comparison_layer` uniform(-1, 1), `wager` uniform(0, 0.1)
>
> **2. `MAPPOActor`** (baseline actor, meta=False) :
> CNN encoder → RNN (GRU) → ACT layer (Categorical)
>
> **3. `MAPPOCritic`** (baseline critic, meta=False) :
> CNN encoder → RNN (GRU) → v_out Linear (centralized state value)
>
> **4. `MAPSActor`** (meta actor, meta=True) :
> Same as MAPPOActor + `MarlSecondOrderNetwork` on wager path. Includes
> student `layer_input` extra Linear projection. Returns wager_logits
> en plus.
>
> **5. `MAPSCritic`** (meta critic, meta=True) :
> Uses `RNNLayerMeta` (GRU + cascade) but **no 2nd-order** (paper Fig.4
> wager only on actor side).
>
> **6. `MAPPOPolicy`** (wrapper) :
> Manage les 4 networks (actor, critic + optional meta versions) + 2
> pairs of optimizers (actor_opt, critic_opt). Only Adam and RangerVA
> supported (E.5 scope lock).
>
> **Methods `MAPPOPolicy`** :
> - `get_actions(obs, rnn_states, masks, available_actions=None,
>   deterministic=False)` → (values, actions, action_log_probs,
>   rnn_states_actor, rnn_states_critic) — used during rollout
> - `evaluate_actions(...)` → (values, action_log_probs, dist_entropy)
>   — used during PPO update
> - `evaluate_actions_meta(...)` → wager_logits — used for MAPS loss
> - `act(...)` — get_actions without value (for evaluation rollouts)
>
> **Optim builder** :
> - `Adam(lr=actor_lr, betas=(0.9, 0.999))` standard
> - `RangerVA(lr=...)` if `torch_optimizer` installed (E.6 — only in
>   `.venv-marl`)

## 2. Contraintes scientifiques

### Paper Fig.4 architecture

```
obs (RGB 11×11×3)
  → CNNBase (Conv16 → FC100 → LayerNorm) → enc (100-d)
  → GRU (hidden=100, recurrent_n=1) → rnn_out
  → [+ MarlSecondOrderNetwork if meta] → wager_logits
  → Categorical(action_space.n) → action
```

Plus critic path :
```
share_obs → CNN → GRU → v_out (Linear → scalar value)
```

### Hidden size 100 (paper Table 12)

Student config utilisait 144 (D-marl-hidden-size). Sprint-09 E.7 aligné
à 100.

### `MAPSCritic` no 2nd-order

Paper Fig.4 montre clairement wager only on actor. Critic has cascade
(via RNNLayerMeta) but no 2nd-order wager head. Important pour parity.

### Init schemes

- `use_orthogonal=True` (default) → `nn.init.orthogonal_`
- `gain=0.01` pour les action heads (sortie petite au début, exploration
  large)
- `gain="relu"` pour les couches intermédiaires
- Biases constant zero

## 3. Contraintes d'ingénierie

### `layer_input` student artifact

Student `MAPSActor` a un `layer_input = nn.Linear(rnn_out, rnn_out)`
extra projection avant le 2nd-order. Pas dans paper Fig.4 mais
kept pour parity student.

### Optim choice : Adam vs RangerVA

Paper Table 12 silent. Student utilise Adam. RangerVA disponible if
deps installed. Notre port supporte les 2, default Adam.

### Separate actor/critic optimizers

Pas un seul optim partagé — actor a son lr (7e-5), critic a son lr
(7e-5 même mais distinct).

### `MAPPOPolicy.use_meta` flag

Construit avec `policy.use_meta = setting.meta`. Si False, skip
construction MarlSecondOrderNetwork. Économise params.

## 4. Deviations / dettes présentes

### D-marl-hidden-size (E.7 resolved)

100 paper-faithful (student 144).

### D-marl-actor-lr, D-marl-critic-lr (E.7 resolved)

7e-5 paper Table 12.

### D-marl-attention-extensions (E.11 policy)

Paper Fig.4 simple : Linear + GRU. Student `modularity.py` a RIM +
SCOFF + skill dynamics + bottom-up attention. **OMITTED** dans port
pour paper-faithful minimal port.

### DETTE-1 cross-ref

`MarlSecondOrderNetwork` est une 3ème variante de SecondOrderNetwork
(après `core.SecondOrderNetwork` Blindsight/AGL et `sarl.SarlSecondOrderNetwork`).
Triplon (DETTE-1 cross-cross-cross-domain).

Unification post-Phase F en `SecondOrderCore` partagé.

### Pas de DETTE active sur ce module spécifiquement

## 5. Questions ouvertes

- **Q1 :** `layer_input` student artifact — vraiment utile ? Ablation :
  removed → meta gain change ?

- **Q2 :** RangerVA disponible mais pas activé par default. Paper
  Table 12 silent. Avec RangerVA, MARL convergerait mieux ?

- **Q3 :** `MAPSCritic` sans 2nd-order — pourquoi pas symétrique avec
  actor ? Paper Fig.4 le dit, mais raison théorique ?

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- 6 classes architecture
- Separate actor/critic optimizers
- `use_meta` flag for memory-saving
- Orthogonal init + gain=0.01 action heads
- `layer_input` student artifact

### À améliorer
- **DETTE-1 unification** : `SecondOrderCore` shared with SARL/CL
  variants
- **Type hints `nn.GRU`** + Protocol
- **Test deterministic action selection**

### Tests à écrire
- Parity bit-exact vs reference
- `use_meta=False` skip 2nd-order construction
- Forward shapes : (B, obs) → (B, actions)
- Init values match (orthogonal scale)
- Optim builder dispatches Adam vs RangerVA

### Connexion
- Utilise : `encoder.CNNBase`, `rnn.{RNNLayer, RNNLayerMeta}`,
  `act.ACTLayer`, `util.init`
- Appelé par : `runner.MeltingpotRunner` (rollout), `trainer.MAPPOTrainer`
  (PPO update)
