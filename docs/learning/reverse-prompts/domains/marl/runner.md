# Reverse-prompt — `src/maps/domains/marl/runner.py`

**Module path actuel (sur main) :** `src/maps/experiments/marl/runner.py`
**Taille :** ~723 lignes.
**Paper :** §B.4, Fig.4.

---

## 1. Quelle aurait été la spec ?

> Écris le rollout + training orchestrator pour MARL (MeltingPot +
> MAPPO + MAPS). Port `external/paper_reference/marl_tmlr/onpolicy/
> runner/separated/meltingpot_runner.py` trimmed.
>
> **Architecture "separated MAPPO"** :
> - 1 `(MAPPOPolicy, MAPPOTrainer, RolloutBuffer)` tuple **par agent**
> - Agents partagent l'env mais train indépendamment
> - Permet de scaler à 10+ agents
>
> **`MeltingpotRunner`** class avec :
> - `__init__(env, policies, trainers, buffers, cfg)`
> - `run()` : outer loop épisodes → collect rollouts → compute returns
>   → PPO updates → log → checkpoint
> - `collect(step)` : env.step + buffer.insert pour tous les agents
> - `compute_wager_objective(rewards)` : EMA + binary target (paper
>   eq.13-14)
>
> **Paper eq.13 + 14 (wager target)** :
> - α=0.45 EMA (paper Table 12 ; student utilise 0.25 — D-marl-ema-alpha
>   fix)
> - `y_wager = (1, 0) if r_t > EMA_t else (0, 1)` (paper eq.14 ;
>   student utilise `grad_rewards > 0` — D-marl-wager-condition fix)
>
> **NaN-guards diagnostiques** (`_buffer_finite_check`,
> `_next_values_finite_check`, `_valuenorm_finite_check`) ajoutés en
> E.16 pour debug. À garder ? Removable post-stabilization.
>
> **Env contract** (à respecter pour `MeltingPotEnv`) :
> - `env.reset() → (obs_dict, info)`
> - `env.step(action_dict) → (obs, rew, done, info)`
> - `observation_space, share_observation_space, action_space` (Dict)
> - `env.close()`

## 2. Contraintes scientifiques

### Pourquoi "separated MAPPO"

Centralized critic with separated actors = standard MAPPO. Mais
"separated" ici veut dire que chaque agent a SON propre critic aussi.
Choix paper.

Avantages : agents peuvent diverger (specializations). Inconvénients :
plus de params, less coordination.

### Paper eq.13-14 (MARL wager)

L'EMA target ressemble à SARL `target_wager` mais en MARL le reward est
un dict (per-agent). La fonction `compute_wager_objective` calcule
l'EMA par agent.

### D-marl-ema-alpha + D-marl-wager-condition (resolved)

Voir deviations.md B.11. Paper α=0.45, student 0.25 → fix paper.
Paper condition `r > EMA`, student `grad_r > 0` → fix paper.

## 3. Contraintes d'ingénierie

### Trimming massif vs student

Student 808 L → port 723 L. Drops :
- Action-env dict flattening (MeltingPot-specific, moved to env.py)
- Dead `get_episode_parameters` helpers
- RLLib integration

### NaN-guards diagnostiques

Phase E.16 a vu des NaN dans le buffer durant smoke runs. Les guards
catch fast avec context :
```
[NaN-guard] buffer.rewards not finite at agent_id=0 stage=after_collect :
NaN=23, Inf=0, shape=(200, 8, 1), min=nan, max=1.5e+03
```

Permet de remonter à la source rapidement.

## 4. Deviations / dettes présentes

### D-marl-cascade-not-implemented (paper-admitted)

Paper Table 12 implicit : MARL n'utilise pas la cascade vraiment
(cascade_iter=1 forcé). Notre port supporte cascade_iter>1 mais ne
l'utilise pas en config par défaut.

### D-marl-ema-alpha (D.2 / E.7 / Sprint-09 9.5 resolved)

α=0.45 paper-faithful.

### D-marl-num-env-steps (E.17b3 resolved)

300k env steps (paper §4 text) vs config legacy 40e6. Aligned.

### Pas de DETTE active

## 5. Questions ouvertes

- **Q1 :** "Separated MAPPO" est non-standard. Most papers use
  shared critic. Pourquoi paper choisit separated ? Trade-off.

- **Q2 :** NaN guards ajoutés en E.16 — should be removed post-
  stabilization ? Garde pour défense ? Discussion.

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Separated MAPPO architecture
- NaN guards diagnostiques
- `compute_wager_objective` per-agent EMA
- Env contract strict

### À améliorer
- **Type hints `MeltingPotEnvProtocol`** plus précis que `Any`
- **Logging structuré per-agent**
- **Async checkpoint** comme SARL

### Tests à écrire
- Smoke run sur mock env
- `compute_wager_objective` matches paper eq.13-14
- NaN guards fire when expected
- Multi-agent rollout shape consistency

### Connexion
- Utilise : `policy.MAPPOPolicy`, `trainer.MAPPOTrainer`, `data.RolloutBuffer`,
  `setting.MarlSetting`, `env.MeltingPotEnv`
- Appelé par : `domains/marl/cli.py`
