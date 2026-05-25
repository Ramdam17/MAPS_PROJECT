# Reverse-prompt — `src/maps/domains/marl/trainer.py`

**Module path actuel (sur main) :** `src/maps/experiments/marl/trainer.py`
**Taille :** ~565 lignes.
**Paper :** §2.2 eq.5 + standard PPO (Yu et al. 2022).

---

## 1. Quelle aurait été la spec ?

> Écris une classe `MAPPOTrainer` (MAPPO PPO updates + MAPS wager
> integration). Port `external/paper_reference/marl_tmlr/onpolicy/
> algorithms/r_mappo/r_mappo.py`.
>
> **Responsabilités** :
> 1. **`cal_value_loss`** : value loss avec PPO clip + optional Huber
>    + optional ValueNorm
> 2. **`ppo_update`** : one mini-batch PPO step — clip policy loss,
>    value loss, entropy bonus, + optional MAPS wager loss (paper
>    eq.5)
> 3. **`train`** : outer loop — GAE advantages → iterate `ppo_epoch`
>    over mini-batches → aggregate train infos
> 4. **`prep_training` / `prep_rollout`** : set modules train/eval mode
>
> **`TrainInfo` dataclass** : value_loss, policy_loss, dist_entropy,
> actor/critic grad_norm, ratio, etc.
>
> **Wager loss** (settings meta=True only) :
> ```python
> wager_loss = F.binary_cross_entropy_with_logits(
>     values_meta, wager_objective
> )
> ```
> où `values_meta` = output 2nd-order (raw logits), `wager_objective`
> = EMA-based target depuis `runner.compute_wager_objective`.
>
> **Dropped vs student** :
> - `FocalLoss` class (dead code)
> - `PopArt` (default `use_popart=False`)
>
> **Optim : ONE wager forward call** au lieu de 2 (student fait
> `evaluate_actions_meta` 2× — pour actor wager + critic wager — mais
> le signal est identique).
>
> **NaN guards** _assert_finite, _assert_weights_finite — E.16 debug.

## 2. Contraintes scientifiques

### Standard PPO clip (Yu et al. 2022)

```
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

Plus entropy bonus `β·H(π)`. Standard.

### Value clip (optional)

PPO value loss peut être clippé (Schulman 2017 paper II). Active via
`use_clipped_value_loss`. Notre port le supporte.

### ValueNorm pre-loss

Si `use_valuenorm=True`, on dénormalize les `value_preds` avant de
compute la loss, puis normalize le `return_target`. Yu et al. 2022
montre que c'est plus stable.

### MAPS wager loss = paper eq.5

`BCE-with-logits(wager_logits[B, 2], wager_objective[B, 2])`. Same
loss qu'en SARL trainer. Cohérent.

## 3. Contraintes d'ingénierie

### `ppo_epoch` iterations

Standard MAPPO : `ppo_epoch=15`. Chaque iteration repasse tous les
mini-batches. Plus d'epochs = plus de gradient steps mais risque
d'overfitting au rollout.

### Mini-batch size

`num_mini_batch` (default 1 = full batch). Trade-off similar à PPO
standard.

### Wager loss optional

Quand `setting.meta=False`, skip la 2nd-order branch entièrement.
Sauf que la 2nd-order network n'est pas constructed dans ce cas
(checked via `if policy.use_meta`).

### Grad norm clipping

`torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),
self.max_grad_norm)`. Standard, evite explosions gradient.

## 4. Deviations / dettes présentes

### D-marl-actor-lr, D-marl-critic-lr, D-marl-entropy-coef (E.7 resolved)

Paper Table 12 :
- `actor_lr = 7e-5` (student shell 2e-5)
- `critic_lr = 7e-5` (paper Table "100" typo)
- `entropy_coef = 0.01` (student shell 0.004)

Tous alignés paper-faithful in `marl.yaml`.

### D-marl-hidden-size (E.7 resolved)

100 paper-faithful (student 144).

### D-marl-cascade-not-implemented (paper-admitted)

Cascade off in factorial default. Code supports si on l'active.

### Pas de DETTE active

## 5. Questions ouvertes

- **Q1 :** Le `evaluate_actions_meta` 2× student → 1× port. Sûr que
  c'est équivalent ? Le second call recompute le wager forward (avec
  dropout potentially redrawn). Si on cache, mask different.

- **Q2 :** PPO + cascade interaction : 50 cascade iters sur RNN +
  PPO mini-batches → boucles imbriquées coûteuses. Worth optim ?

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- `cal_value_loss` clipped + Huber + ValueNorm
- `ppo_update` complete PPO step
- One-call optim pour wager forward
- NaN guards
- Drop FocalLoss + PopArt
- Skip wager branch quand meta=False

### À améliorer
- **Documenter explicitement** le pattern "two evaluate_actions_meta
  → one"
- **Pydantic schema** pour TrainInfo
- **Type hints stricts**

### Tests à écrire
- `cal_value_loss` matches reference
- `ppo_update` matches reference
- Wager loss only when meta=True
- Grad norm clipping fires when expected

### Connexion
- Utilise : `policy.MAPPOPolicy`, `valuenorm.ValueNorm`, `util.huber_loss`,
  `mse_loss`, `get_grad_norm`
- Appelé par : `runner.MeltingpotRunner.train`
