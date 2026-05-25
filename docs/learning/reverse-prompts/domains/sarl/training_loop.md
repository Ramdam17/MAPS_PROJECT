# Reverse-prompt — `src/maps/domains/sarl/training_loop.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl/training_loop.py`
**Taille :** 933 lignes — le plus gros module SARL.
**Paper :** §3, Table 11 (hyperparams).
**Review existant :** `docs/reviews/sarl-training-loop.md`.

---

## 1. Quelle aurait été la spec ?

> Écris le **training orchestrator** pour SARL. Porte la training-loop
> half de `external/MinAtar/examples/maps.py:1095-2145` (la `dqn()`
> function, **standard DQN path only** — la continuous-learning branch
> vit dans `sarl_cl`).
>
> **Scope (in/out)** :
> - IN : init networks (v1 ou v2), optimizers, schedulers, replay buffer ;
>   outer loop episode → env-step → buffer-add → ε-greedy → update via
>   `sarl_update_step` ; target-net sync every N updates ; validation
>   rollouts every M episodes
> - OUT : continual learning / teacher / distillation (→ `sarl_cl`)
> - OUT : resume from paper-format checkpoint (~20 keys)
> - OUT : plotting / z-score (→ scripts/notebooks)
>
> **Parity boundary** : the **inner update** (`sarl_update_step`) est
> Tier-3 bit-exact. The **outer loop** est structurally faithful mais
> **PAS** bit-exact (ε-greedy RNG consumption depends on greedy branch
> count, which depends on weights, which depends on prior numerical
> drift).
>
> **Paper constants** (source `maps.py:92-107`, Table 11) :
> - `BATCH_SIZE = 128`
> - `REPLAY_BUFFER_SIZE = 100_000`
> - `REPLAY_START_SIZE = 5_000`
> - `TRAINING_FREQ = 1` (update every N frames once warm)
> - `TARGET_NETWORK_UPDATE_FREQ = 1_000`
> - `STEP_SIZE_1 = 0.0003` (1st-order lr)
> - `STEP_SIZE_2 = 0.0002` (2nd-order lr) — D-sarl-lr-2nd D.9
> - `ADAM_BETAS = (0.95, 0.95)` — D-sarl-adam-beta1/2 D.9
> - `SCHEDULER_STEP = 0.999` (gamma)
> - `SCHEDULER_PERIOD = 1` — D-sarl-sched-step D.9 (paper-as-written
>   typo suspected)
> - `gamma = 0.999` (DQN discount) — D-sarl-gamma D.7
>
> **6 paper settings → `cascade_iterations_1, cascade_iterations_2,
> meta`** mapping :
> 1. (1, 1, False) — vanilla DQN
> 2. (50, 1, False) — cascade on FO only
> 3. (1, 1, True) — meta on, cascade off
> 4. (50, 1, True) — meta + cascade on FO
> 5. (1, 50, True) — meta + cascade on SO
> 6. (50, 50, True) — full MAPS
>
> **Variant selector `_MODEL_VARIANTS = {"v1": (...), "v2": (...)}`** :
> dispatches sur `cfg.model_variant`. Default `"v1"` paper-canonical
> (Sprint 09).
>
> **Dataclass `SarlTrainingConfig`** : 30+ knobs (game, seed, num_frames,
> batch_size, replay sizes, lrs, scheduler, gamma, alpha EMA,
> first_order_loss_kind, model_variant, validation cadence, checkpoint
> cadence, resume_from, device, output_dir).
>
> **Dataclass `TrainingMetrics`** : per-episode returns/lengths/frames/
> losses + validation summaries + total_updates/frames + wall_time +
> cascade_effective_iters (logged for D-sarl-cascade-noop traceability).
>
> **`_check_first_order_loss_kind(kind)`** : fail-fast guard. Raise
> NotImplementedError si `kind="simclr"` (D-002 stub).
>
> **Checkpoint mechanism** (D.13) : every N updates, save `checkpoint.pt`
> avec full state (policy_net, target_net, second_order, optimizers,
> schedulers, replay_buffer, cfg snapshot, RNG states). Resume via
> `cfg.resume_from = Path(...)`.

## 2. Contraintes scientifiques

### 6 settings = 2×2×2 factorial moins legacy

Combinatoire formelle : `cascade_1st × cascade_2nd × meta = 8`.
Paper utilise 7 (1 + 2 + 3 + 4 + 5 + 6 + 7=ACB), pas 8.

Setting 7 = ACB (Actor-Critic Baseline, Young & Tian 2019) — algorithme
**structurellement différent** (pas de cascade ni 2nd-order). Vit dans
`actor_critic.py`.

### `gamma = 0.999` vs 0.99 student

Paper Table 11 dit 0.999 (very long-horizon, valeurs lointaines
comptent). Student `sarl_maps.py:104` utilisait 0.99 (plus standard
DQN). Sprint-08 D.7 a aligné à 0.999 par défaut.

Override pour reproduire student : `--override training.gamma=0.99`.

### `STEP_SIZE_2 = 0.0002`

Paper Table 11. Student `step_size2 = 0.00005` (4× plus petit). D.9
aligné à paper. Override `-o optimizer.lr_second_order=0.00005`.

### `ADAM_BETAS = (0.95, 0.95)`

Paper Table 11. Standard PyTorch défault (0.9, 0.999). Pas student
explicit. D.9 aligné à paper. Override
`-o optimizer.betas=[0.9,0.999]`.

### `SCHEDULER_PERIOD = 1`

Paper Table 11 dit `step_size=1`. Mais avec `StepLR(step_size=1,
gamma=0.999)` → décay 0.999× chaque update → vite près de zéro. **Suspected
paper typo** (probably meant 1000). Sprint-08 D.9 garde `step_size=1`
paper-as-written + emit `log.warning` quand step_size=1. Override
`-o scheduler.step_size=1000` pour student.

## 3. Contraintes d'ingénierie

### Outer loop structurally faithful

L'outer loop suit l'ordre paper exactement :
1. Reset env, get state
2. For each frame :
   - ε-greedy action (using `epsilon_greedy_action` from `rollout.py`)
   - env.act(action) → (reward, done)
   - buffer.add(transition)
   - If buffer warm AND t % training_freq == 0 :
     - sample = buffer.sample(B)
     - sarl_update_step(sample, ...)
   - If t % target_update_freq == 0 :
     - target_net.load_state_dict(policy_net.state_dict())
3. End episode → record metrics
4. Every validation_every_episodes : run greedy validation

**Pas bit-exact** car ε-greedy RNG consumption drift inevitable.

### Checkpoint cadence

`checkpoint_every_updates` (default 10_000). Save tout l'état. Permet
de reprendre un long run interrupted. Pour 500_000 frames = 50
checkpoints. ~MB chacun.

### `cascade_effective_iters` traceability

Quand `meta=False` + cascade ON sur 1st-order, le cascade est no-op
(D-sarl-cascade-noop). On log `cascade_effective_iters_1=1` dans
metrics.json pour traçabilité. Permet aux aggregate scripts de
flagger les settings affectés.

### Resume schema versioning

Le checkpoint contient un snapshot du `cfg` complet. Au load, on
compare avec le `cfg` actuel — mismatch sur game/seed/meta/cascade/
num_frames raise ValueError. Évite les bugs subtils de "resume avec
cfg différent".

### MinAtarLike Protocol

Typed Protocol définissant le contract env :
`reset, state, act, num_actions, state_shape`. Permet tests sans
MinAtar installé (mock env).

## 4. Deviations / dettes présentes

### 🚨 D-sarl-wrong-variant (Sprint 09 resolved via model_variant)

`cfg.model_variant` selector + `_MODEL_VARIANTS` dispatch. Default
`"v1"` paper-canonical. Override `"v2"` pour archived Phase F.4 runs.

### ⚠️ Plusieurs D-sarl-* concernent les paper constants (résolus D.7/D.9/D.12)

- D-sarl-num-frames : 500_000 paper Table 11 (override 5_000_000 pour
  student)
- D-sarl-lr-2nd : 0.0002 paper (override 0.00005)
- D-sarl-adam-beta1/2 : (0.95, 0.95) paper (override (0.9, 0.999))
- D-sarl-gamma : 0.999 paper (override 0.99)
- D-sarl-sched-step : 1 paper-as-written (override 1000)

Tous overridable via `-o` CLI. Defaults paper-faithful.

### ⚠️ D-sarl-alpha-ema (D.2 resolved)

α=0.45 paper Table 11 (override 0.25 pour student `-ema 25`). Géré
dans `cfg.alpha` (en percent, divisé par 100 dans `target_wager`).

### ⚠️ D-sarl-target-update (D.7 doc-only)

`TARGET_NETWORK_UPDATE_FREQ = 1000` paper Table 11. Student `maps.py`
utilisait 100 dans la fonction `dqn()` inline (L1188, 1193). Notre port
paper-faithful = 1000.

### Pas de DETTE active

Module gros (933 L) mais propre. Bien factorisé via `_build_networks`,
`_build_optimizers`, helpers.

## 5. Questions ouvertes

- **Q1 :** Le `SCHEDULER_PERIOD = 1` (paper) est-il vraiment un typo ?
  Avec `gamma=0.999, step=1` après 500k updates → `0.999^500000 ≈ 0`.
  Donc la LR tombe à zéro très vite, optimizer arrête de bouger.
  Empirically c'est OK ? Tests devraient compare avec `step=1000`.

- **Q2 :** Pourquoi `gamma=0.999` (paper) vs 0.99 (student) ? 0.999
  donne un horizon effectif ~1000 steps, 0.99 ~100 steps. MinAtar
  games ont des épisodes ~100-500 steps. 0.999 = un peu trop long
  potentiellement.

- **Q3 :** `ADAM_BETAS=(0.95, 0.95)` — paper diverge des defaults
  (0.9, 0.999). Faible β2 = adapté aux gradients très variables
  (DQN). Sens.

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- `SarlTrainingConfig` dataclass complet
- `setting_to_config(N, base)` 6-setting mapping
- `_MODEL_VARIANTS` v1/v2 dispatch
- Paper constants comme module-level (override via CLI)
- Checkpoint mechanism avec schema versioning
- `MinAtarLike` Protocol pour tests
- Validation cadence + greedy validation (no ε)
- Fail-fast SimCLR guard

### À améliorer
- **Schedule period typo** : si tests confirment que `step=1000` est
  meilleur, switch default + override pour paper-as-written. Sinon
  garder warning.
- **TensorBoard / wandb logging** : actuellement metrics.json en fin
  de run. Live logging serait utile pour long runs.
- **Async checkpoint** : `torch.save` est blocking. Pour des
  checkpoints fréquents, async I/O thread.
- **GPU memory tracking** : MARL-style énergie tracking ici pour
  SARL aussi ?
- **Type hints `MinAtarLike`** : la Protocol est minimale, pourrait
  ajouter `lives()`, `episode_count()`.

### Tests à écrire
- Smoke : `run_training(cfg(num_frames=1000))` finit sans crash
- Setting → config mapping correct (6 settings)
- Variant v1 vs v2 dispatch (different network types)
- Checkpoint save / load roundtrip
- Resume from checkpoint avec compatible cfg
- Resume avec mismatched cfg raise ValueError
- Greedy validation no ε (deterministic)

### Opportunités perf identifiées
- **Cascade no-op skip** : si `meta=False`, `cascade_iterations_1=50`
  donne identique à `=1`. Détecter et skip → 50× speedup setting 2.
- **target_wager vectorisation** (DEJA dans `data.py` mais reste un
  bottleneck 32% wall setting 1)
- **GPU pour MinAtar env** : actuellement env step CPU only.
  MinAtar Python = slow. Pas trivial à porter GPU.
- **Replay buffer in shared memory** : pour multi-seed parallel
  local runs

### Connexion
- Utilise : `data` (replay + state), `losses` (cae_loss), `model` (v2),
  `model_v1` (v1), `trainer.sarl_update_step`, `rollout.epsilon_greedy_action`,
  `evaluate.aggregate_validation`
- Configure via `config/domains/sarl/training.yaml`
- Appelé par `domains/sarl/cli.py`

## Méta — pour Claude chat

```text
J'orchestre un training DQN avec MAPS components sur MinAtar (paper
Vargas 2025). Le training loop fait :

1. Outer loop episode → env.step → replay buffer
2. Once warm, every N frames : sample batch, call inner update step
3. Every M episodes : greedy validation rollouts

L'inner update est Tier-3 bit-exact vs reference (10⁻⁶ tolerance).
L'outer loop est structurellement fidèle mais PAS bit-exact (ε-greedy
RNG drift inevitable car action choice depends on weights, which
depend on cumulative numerical drift).

Question : pour reproduire des Tables paper avec N=3 seeds, est-ce
qu'un outer loop non bit-exact suffit ? Mon hypothèse : oui car la
moyenne sur 3 seeds doit dominer la variance individuelle. Mais
comment quantifier formellement la "robustesse" de cette claim ?

Faut-il :
- Plus de seeds (10-30) pour absorber le drift outer loop ?
- Tier-4 parity (outer loop bit-exact) en désactivant ε-greedy
  exploration ?
- Statistical tests style Henderson 2018 "deep RL that matters" ?
```
