# Reverse-prompt — `src/maps/domains/marl/env.py`

**Module path actuel (sur main) :** `src/maps/experiments/marl/env.py`
**Taille :** ~481 lignes.
**Linux-only** (dmlab2d, meltingpot deps).

---

## 1. Quelle aurait été la spec ?

> Écris les wrappers MeltingPot pour MARL. Port
> `external/paper_reference/marl_tmlr/onpolicy/envs/meltingpot/
> MeltingPot_Env.py` trimmed.
>
> **Composants** :
>
> **1. `spec_to_space(spec)`** : `dm_env.specs` → gymnasium.spaces.
> Handles Discrete, BoundedArray, Array, Tuple, Dict recursive.
>
> **2. `timestep_to_observations(timestep)`** : extract per-player
> dict restricted to `{"RGB", "WORLD.RGB"}` (paper Fig.4).
>
> **3. `MeltingPotEnv`** : gymnasium-style multi-agent wrapper over
> dmlab2d env. Contract :
> - `reset() → (obs_dict, info)`
> - `step(action_dict) → (obs, rew, done, info)`
> - `observation_space, share_observation_space, action_space` (Dict
>   with `"player_i"` keys)
> - `close()`
>
> **4. `downsample_observation(obs, factor=8)`** : scale RGB substrate
> from 8x sprites to 1x (88×88 → 11×11). Paper §A.4.
>
> **5. `DownSamplingSubstrateWrapper`** : applique downsample sur each
> player obs.
>
> **6. `env_creator(substrate, roles)`** : substrate_id + roles list →
> dmlab2d env → DownSample wrapper → MeltingPotEnv.
>
> **Simplifications vs student** :
> - Drop `ray.rllib.MultiAgentEnv` inheritance (pas besoin de RLLib)
> - Drop `DataExtractor` debug helper (dev-only)
> - Drop `n_rollout_threads > 1` support (1 thread = 1 step)
>
> **Lazy import** : `dmlab2d`, `meltingpot`, `cv2` importés à
> l'intérieur des constructors/factories. Module importable depuis
> main venv (3.12) sans ces deps ; les helpers purs marchent avec mock.

## 2. Contraintes scientifiques

### MeltingPot 2.0 substrates

4 substrates testés :
- **commons_harvest_closed** : 10 agents, tragedy of commons closed
- **commons_harvest_partnership** : 10 agents, partnership variant
- **chemistry** : variable agents, branched reactions
- **territory_inside_out** : variable agents, territorial dynamics

Tous testés paper Table 7.

### Paper §A.4 — observation downsample

Original substrate = 88×88×3 (8x sprite size). Pour réduire compute,
downsample à 11×11×3. Quality loss acceptable per paper.

### `RGB` vs `WORLD.RGB`

- `RGB` : observation du player (POV)
- `WORLD.RGB` : observation centrale (full state, pour centralized
  critic)

Paper utilise les 2. Centralized critic prend `WORLD.RGB`, actor
prend `RGB`.

## 3. Contraintes d'ingénierie

### Lazy imports critical

dmlab2d / meltingpot ne sont pas dispo dans main venv (Python 3.12).
Lazy import permet tests + import depuis 3.12. Construct uniquement
dans `.venv-marl` (3.11).

### `n_rollout_threads = 1` only

Student supportait multi-thread rollouts via parallel envs. Notre port
strip ça (simplifier). 1 thread per process. Multi-process via SLURM
array.

### `_OBSERVATION_PREFIX` whitelist

Filter les observations à ce qu'on consomme effectivement (`RGB`,
`WORLD.RGB`). MeltingPot expose 20+ keys, on en garde 2.

### Gymnasium contract

`reset` retourne `(obs, info)`, `step` retourne `(obs, rew, done,
info)`. Compatible avec gymnasium 0.29+.

## 4. Deviations / dettes présentes

### Pas de déviation paper sur ce module

Wrapping pur, pas de math. Match student.

### Linux-only constraint

dmlab2d ne build pas sur macOS (native deps). Ce module est unusable
en dev local Mac. SLURM only.

**Dette potentielle** : si Rémy veut tester MARL en local, il faudrait
soit Docker Linux soit mock env. Currently impossible.

### Pas de DETTE active code-wise

## 5. Questions ouvertes

- **Q1 :** Peut-on faire un MARL "lite" sans MeltingPot pour dev
  local ? Par ex. avec PettingZoo simple environments (Atari, MPE).
  Permettrait d'iterer rapide sur le port avant SLURM.

- **Q2 :** Le downsample 11×11 perd beaucoup d'info visuelle. Paper
  ne discute pas du trade-off accuracy vs speed. Ablation ?

- **Q3 :** `WORLD.RGB` centralized critic — fair if all agents
  voient le state global ? Compétitif si certains agents partial
  observabiliy ?

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Lazy imports (compat 3.12)
- Filter `RGB` + `WORLD.RGB` only
- DownSample 11×11 default
- Gymnasium contract

### À améliorer
- **Mock env pour Mac dev** : permettre `--substrate mock` qui spawn
  un fake MeltingPot pour iteration locale
- **Multi-thread rollout** (futur, post-paper)
- **Type hints `dm_env.TimeStep`** plus précis

### Tests à écrire
- `spec_to_space` Discrete, Box, Tuple, Dict
- `timestep_to_observations` filter RGB/WORLD.RGB
- `downsample_observation` 88×88 → 11×11
- Mock env satisfait le contract

### Connexion
- Utilisé par : `runner.MeltingpotRunner`, `cli.run_marl`
- Dépend de : dmlab2d, meltingpot (Linux only)
