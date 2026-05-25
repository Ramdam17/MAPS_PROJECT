# Reverse-prompt — `src/maps/domains/marl/data.py`

**Module path actuel :** `src/maps/experiments/marl/data.py`
**Taille :** ~340 lignes.
**Paper :** Yu et al. 2022 (PPO + GAE).

---

## 1. Quelle aurait été la spec ?

> Écris une class `RolloutBuffer` per-agent pour separated MAPPO.
> Port `external/paper_reference/marl_tmlr/onpolicy/utils/
> separated_buffer.py:SeparatedReplayBuffer` trimmed.
>
> **Per-agent buffer** holding one agent's rollout across
> `episode_length` steps and `n_rollout_threads` parallel envs.
>
> **Storage arrays** (numpy) :
> - `obs[step, thread, *obs_shape]`
> - `share_obs[step, thread, *share_obs_shape]`
> - `rewards[step, thread, 1]`
> - `actions[step, thread, *act_shape]`
> - `action_log_probs[step, thread, 1]`
> - `value_preds[step+1, thread, 1]`
> - `returns[step+1, thread, 1]`
> - `advantages[step, thread, 1]` (computed)
> - `masks[step+1, thread, 1]` (0 if terminal)
> - `active_masks[step+1, thread, 1]` (0 if agent dead)
> - `rnn_states[step+1, thread, recurrent_n, hidden]`
> - `rnn_states_critic[...]`
>
> **API** :
> - `__init__(episode_length, n_rollout_threads, hidden_size,
>   recurrent_n, gamma, gae_lambda, obs_space, share_obs_space,
>   act_space, use_valuenorm)`
> - `insert(obs, share_obs, rnn_states, rnn_states_critic, actions,
>   action_log_probs, value_preds, rewards, masks, active_masks)`
> - `after_update()` : copy last step → first step (for next rollout)
> - `compute_returns(next_value, value_normalizer)` : GAE backward
>   pass
> - `feed_forward_generator(advantages, num_mini_batch, mini_batch_size=None)`
> - `recurrent_generator(advantages, num_mini_batch, data_chunk_length)`
>
> **GAE formula** :
> ```
> δ_t = r_t + γ·V(s_{t+1})·mask_{t+1} - V(s_t)
> A_t = δ_t + γ·λ·mask_{t+1}·A_{t+1}
> ```
>
> **Dropped vs student** :
> - LSTM (`rnn_cells`) — GRU only
> - `naive_recurrent_generator` (not used)
> - `chooseinsert` / `chooseafter_update` (SC2-specific)
> - `bad_masks` (proper_time_limits off)
> - `store_action_and_rnn_state` (MI logging off)

## 2. Contraintes scientifiques

### Generalized Advantage Estimation (GAE)

Schulman 2015. Formule :
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t^GAE(λ) = Σ_{k=0}^∞ (γλ)^k · δ_{t+k}
```

Trade-off bias-variance via λ :
- λ=1 → Monte Carlo (variance high, bias zero)
- λ=0 → TD(0) (variance low, bias high)
- λ=0.95 standard (paper)

### ValueNorm interaction

Si `use_valuenorm=True`, on dénormalize les `value_preds` avant le
backward GAE pass (parce que les rewards sont natural-scale, pas
normalized). Puis normalize les returns avant value loss compute.

### Per-agent vs shared buffer

Separated MAPPO = per-agent buffers. Une instance par agent. Agents
training indépendants.

## 3. Contraintes d'ingénierie

### `_flatten(T, N, x)`

Helper utility : (T, N, *) → (T·N, *). Used dans les mini-batch
generators pour flatten time + thread dims.

### Mini-batch generators

2 types :
- `feed_forward_generator` : random shuffle, batch=mini_batch_size or
  T·N/num_mini_batch. Standard PPO.
- `recurrent_generator` : chunks of `data_chunk_length` (=10 typique)
  preserved temporellement, shuffle chunks. Pour PPO recurrent.

Notre port utilise recurrent_generator (RNN-based policies).

### Memory cost

`episode_length=200, n_rollout_threads=16, obs=11×11×3, hidden=100`
:
- `obs` : 200 × 16 × 363 × 4 bytes = ~4.6 MB per buffer
- `rnn_states` : 200 × 16 × 1 × 100 × 4 = ~1.3 MB
- Total per agent : ~15 MB
- × 10 agents = 150 MB. OK.

### `after_update()` copy

Important : after each PPO update, le buffer's last step (`buffer[T]`)
devient le first step du prochain rollout. Prevents discontinuity.

## 4. Deviations / dettes présentes

Pas de déviation paper sur ce module.

**Pas de DETTE active.**

## 5. Questions ouvertes

- **Q1 :** GAE λ=0.95 vs 0.99 ? Paper Table 12 dit 0.95. Standard.

- **Q2 :** `recurrent_generator` chunks=10 — bon trade-off ? Plus
  long = plus de contexte mais moins de chunks (variance estim higher).
  Plus court = moins de contexte.

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Per-agent storage
- GAE backward pass
- ValueNorm interaction
- recurrent_generator avec chunks
- `after_update()` copy

### À améliorer
- **Type hints `numpy.ndarray`** plus précis
- **Schema versioning** pour le checkpoint resume

### Tests à écrire
- `compute_returns` matches reference GAE
- ValueNorm interaction correct
- `feed_forward_generator` shapes correct
- `recurrent_generator` chunks preserve temporality
- `after_update` copy

### Opportunités perf
- **Pre-allocate arrays** : already done numpy.zeros
- **Vectoriser GAE backward** : actuellement loop Python. Peut-être
  cumsum-based vectorisation.

### Connexion
- Utilisé par : `runner.MeltingpotRunner` (collect/store), `trainer.MAPPOTrainer`
  (sample batches)
