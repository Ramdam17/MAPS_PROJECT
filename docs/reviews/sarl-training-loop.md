# Component review — `src/maps/experiments/sarl/training_loop.py`

**Review sub-phase :** Sprint-08 D.8.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl/training_loop.py` (512 L, 2 dataclasses
+ 3 helpers + main entrypoint + setting mapping).
**Paper sources :** Table 11 (SARL hyperparams), §3 (SARL training loop). Paper settings 1-6.
**Student sources :** `external/paper_reference/sarl_maps.py:1095-1930` (`dqn()` + main loop
+ validation).

**Callers du port :**
- `scripts/run_sarl.py:188` (CLI entry point, Sprint-07).
- SLURM batch Sprint-07 (900 runs via `sbatch --array`).
- Tests unit (pas d'appel direct, via fixtures).

**DoD D.8** : doc créé, 6 sections (constants, configs, build_*, run_training, persist, setting),
0 code touché, toutes les D-sarl-* cross-ref'd.

---

## (a) Module-level paper constants (L74-83)

Port module constants :

```python
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100_000
REPLAY_START_SIZE = 5_000
TRAINING_FREQ = 1
TARGET_NETWORK_UPDATE_FREQ = 1_000
MIN_SQUARED_GRAD = 0.01
STEP_SIZE_1 = 0.0003
STEP_SIZE_2 = 0.00005
SCHEDULER_STEP = 0.999
SCHEDULER_PERIOD = 1_000
```

### Audit vs paper Table 11 + student

| Const               | Paper Table 11 | Student `sarl_maps.py` | Port                    | Verdict |
|:--------------------|:--------------:|:-----------------------|:------------------------|:------:|
| `BATCH_SIZE`        | **128** (row 1) | 128 (L99)             | 128                     | ✅ paper-faithful (D.12 correction — earlier D.8 grep claim "32" was a misread of paper_tables_extracted.md) |
| `REPLAY_BUFFER_SIZE`| 100,000        | 100,000                | 100,000                 | ✅    |
| `REPLAY_START_SIZE` | 5,000          | 5,000                  | 5,000                   | ✅    |
| `TRAINING_FREQ`     | implicit "every frame" | 1                | 1                       | ✅    |
| `TARGET_NETWORK_UPDATE_FREQ` | 1,000 | **100** (L1188,1193)    | 1,000                   | 🆘 port=paper (student was wrong) — D-sarl-target-update resolved ✅ |
| `MIN_SQUARED_GRAD`  | 0.01           | 0.01                   | 0.01                    | ✅    |
| `STEP_SIZE_1`       | **0.0003** (row 9) | 0.0003 (L98)       | 0.0003                  | ✅ paper-faithful (D.12 correction — earlier D.8 claim "0.00025" was a misread) |
| `STEP_SIZE_2`       | 0.0002         | **0.00005** (student wrong) | 0.00005            | 🆘+❌ D-sarl-lr-2nd (already tracked, queued D.9) |
| `SCHEDULER_PERIOD`  | **1**          | 1,000 (student wrong)  | 1,000                   | 🆘+❌ D-sarl-sched-step (already tracked, queued D.9) |
| `SCHEDULER_STEP`    | silent         | 0.999                  | 0.999                   | ⚠️ D-sarl-sched-gamma (info) |

### D.8-F1 and D.8-F2 — **RETRACTED 2026-04-20 in D.12**

The D.8 review originally flagged `BATCH_SIZE=128` and `STEP_SIZE_1=0.0003` as port-only
divergences from paper (claimed paper=32 / 0.00025). Both claims were wrong — a misreading of
`paper_tables_extracted.md` that contradicts the source-of-truth `paper_vs_code_audit.md` rows 1
and 9. Paper Table 11 actually specifies **128** and **0.0003**; student matches; port was
always paper-faithful on these two parameters. D.9 unfortunately applied the proposed "fixes"
and introduced regressions; D.12 reverts them. **No deviation exists for either parameter.**
See commit log for the D.12 revert + the honest retraction of D8-F1/D8-F2 findings.

## (b) `SarlTrainingConfig` dataclass (L99-151)

Fields principaux :
- Env : `game`, `seed`.
- Setting-derived : `meta`, `cascade_iterations_1`, `cascade_iterations_2`.
- DQN : 9 fields (num_frames, batch, replay_*, training_freq, target_update_freq, step_sizes,
  scheduler_*, gamma).
- EMA wagering : `alpha`.
- Validation : `validation_every_episodes`, `validation_iterations`.
- Runtime : `device`, `output_dir`.

### 🚨 D8-F3 — `alpha: float = 1.0` default obsolete post-D.2

Post-D.2 (2026-04-20) le config yaml a `alpha: 45` (paper-faithful). Le dataclass default L143 est
resté à `1.0` (l'ancienne valeur). Le CLI `run_sarl.py` charge le yaml → override le default →
en prod `alpha = 45`. **Mais** tout caller qui instancie `SarlTrainingConfig()` directement
(tests ?) récupère `1.0` → diverge de la config prod.

**Piste D8-fix-3** : aligner default `alpha: float = 45.0`. Zero impact prod (yaml override), +1
safety net pour instanciations programmatiques.

### ✅ Autres fields cohérents
- `gamma = 0.999` post-D.7 ✅.
- `num_frames = 5_000_000` — 🆘 D-sarl-num-frames (paper 500k/1M selon interprétation). Queued D.12.
- Autres cohérents avec module constants (même fonctionnement que `STEP_SIZE_1 = 0.0003` issue).

## (c) `TrainingMetrics` dataclass (L154-170)

Post-D.4 modifié avec `cascade_effective_iters_1: int = 1` et `cascade_effective_iters_2: int | None`.

✅ **Parity + augmentation** : fields originaux student + 2 fields diagnostics port.

## (d) `_build_networks` (L180-198)

```python
torch.manual_seed(cfg.seed)
policy = SarlQNetwork(in_channels, num_actions).to(cfg.device)
target = SarlQNetwork(in_channels, num_actions).to(cfg.device)
target.load_state_dict(policy.state_dict())
target.eval()
second = SarlSecondOrderNetwork(in_channels).to(cfg.device) if cfg.meta else None
```

Vs student `sarl_maps.py:1226-1268` : même init order (policy → target → second), même
`load_state_dict(policy)` pour target, même `.eval()` target.

✅ **Bit-parity init**.

⚠️ Note post-D.7 : port `SarlQNetwork` inclut maintenant `b_recon` zero-init. Teacher
`target.load_state_dict(policy.state_dict())` copie aussi `b_recon=0` → target a b_recon aussi.
Consistent.

Local `torch.manual_seed(cfg.seed)` — doc C.15 clarifie "re-seed local pour init net
déterministe". Upstream caller responsible pour Python random + NumPy seeding via
`set_all_seeds`.

## (e) `_build_optimizers` (L201-215)

```python
opt1 = optim.Adam(policy.parameters(), lr=cfg.step_size_1, eps=MIN_SQUARED_GRAD)
sch1 = StepLR(opt1, step_size=cfg.scheduler_period, gamma=cfg.scheduler_gamma)
opt2 = optim.Adam(second.parameters(), lr=cfg.step_size_2, eps=MIN_SQUARED_GRAD) if meta else None
sch2 = StepLR(opt2, step_size=cfg.scheduler_period, gamma=cfg.scheduler_gamma) if meta else None
```

### 🚨 D-sarl-adam-beta1/2 — Adam betas absent

Port **n'utilise pas** `betas=...` → PyTorch default `(0.9, 0.999)`. Paper Table 11 prescrit
`(0.95, 0.95)`. Student ne passe pas non plus `betas` → default `(0.9, 0.999)` aussi.

→ **Port = student** sur betas par défaut, mais **tous deux divergent paper**. 🆘+❌
D-sarl-adam-beta1/2 (already tracked, queued D.9).

**Piste D.9 fix** : `optim.Adam(..., betas=(0.95, 0.95))` en paper-faithful mode, config-driven.

## (f) `run_training` main loop (L221-413) — **cœur**

### Structure
1. Setup : device, state_shape, num_actions.
2. Log INFO + warning D.4.
3. `_build_networks` + `_build_optimizers` + buffer.
4. Metrics + cascade_effective_iters record.
5. Outer `while t < cfg.num_frames` : episode loop.
6. Inner `while not done` : frame loop.
7. Action selection via `epsilon_greedy_action`.
8. Env step → buffer.add.
9. Update conditionnelle (`t > replay_start_size AND len(buffer) >= batch_size AND t % training_freq == 0`).
10. Target-net sync sur **update-counter clock** (every 1000 updates).
11. Validation cadence (every 50 episodes).
12. Episode bookkeeping + log.

### Parity student `sarl_maps.py:1550-1770`

Port follows the student structure **step-by-step**. Key bit-exact behaviours :

| Aspect                                  | Student                              | Port                                | Match |
|:----------------------------------------|:-------------------------------------|:------------------------------------|:-----:|
| Action via `world_dynamics`             | inline action selection              | `epsilon_greedy_action` (D.5)       | ✅ (split clean) |
| Transition add post-action              | `r_buffer.add(s, s_prime, action, ...)` | idem                             | ✅    |
| Update condition                        | `t > REPLAY_START_SIZE and len >= BATCH_SIZE` | idem L314-318              | ✅    |
| Target sync clock                       | **update-counter** (policy_update_counter % 1000) | idem L345          | ✅    |
| Validation cadence                      | **episode-clock** (every 50 eps)     | idem L384                           | ✅    |

✅ **Main loop bit-parity avec student.**

### D.4 log.warning + metrics D.4 ✅ ajoutés (L252-258, L269-270).

### 🆘 D-sarl-num-frames — `cfg.num_frames = 5_000_000` vs paper

Paper text p.?? : "500,000 frames per game" (alternative : text mentions 1M plus loin, ambigu).
Port default 5,000,000 = **5-10× plus long** que paper. D-sarl-num-frames tracé, queued **D.12**.

**Impact** : si paper z-scores sont atteints à 500k/1M, courir 5M gaspille ~90% du compute.
Inversement, si paper wurde mal transcrit et vraie valeur = 5M (comme student), alignement OK.

**Décision queued D.12** : cross-check text paper + Table 11 + Sprint-07 empirical runs.

## (g) `_persist_outputs` (L421-473)

Save `metrics.json` + `policy.pt` + `second.pt` (si meta) dans `cfg.output_dir`.

Payload JSON inclut (post-D.4) :
- cfg fields (game, seed, meta, cascade_iterations_1/2, num_frames).
- `cascade_effective_iters_1/2` (D.4 artifact pour audit post-hoc).
- `episode_returns/lengths/frames`.
- `episode_losses_first/second`.
- `validation_frames/summaries`.
- `total_updates/frames`, `wall_time_seconds`.

✅ **Artefact riche** pour analyse offline + D.4 audit trail.

## (h) `_SETTING_TABLE` + `setting_to_config` (L479-512)

```python
_SETTING_TABLE = {
    1: (False,  1,  1),   # vanilla DQN
    2: (False, 50,  1),   # cascade on FO only
    3: (True,   1,  1),   # meta on, cascade off
    4: (True,  50,  1),   # meta + cascade on FO
    5: (True,   1, 50),   # meta + cascade on SO
    6: (True,  50, 50),   # full MAPS
}
```

Cross-check paper §3 "6 settings" table : ✅ bit-exact.

`setting_to_config` : applique les 3 fields au-dessus du base config. Si caller passe `base`
avec d'autres overrides (via `-o`), ils sont préservés. **Setting est appliqué LAST** — overrides
CLI ne peuvent pas écraser `meta/cascade_*` qui sont setting-derived.

Port comment `scripts/run_sarl.py:89-93` documente : *"Setting (1-6) is applied LAST so it overrides
any meta/cascade_* values that slipped into the YAML by mistake"*. ✅ **Clean design.**

## Fixes identifiées D.8

| ID          | Fix                                                                       | Scope                                     | Effort |
|:------------|---------------------------------------------------------------------------|-------------------------------------------|:------:|
| D8-fix-1 🚨 | `BATCH_SIZE = 32` (paper-faithful, both paper + student match 32)         | `training_loop.py:74` + config yaml       | queued D.9 |
| D8-fix-2 🚨 | `STEP_SIZE_1 = 0.00025` (paper-faithful, fix 0.0003 port typo)            | `training_loop.py:80` + config yaml       | queued D.9 |
| D8-fix-3    | `alpha: float = 45.0` default (align avec config yaml post-D.2)           | `training_loop.py:143`                    | queued D.9 |
| D8→D.9      | `STEP_SIZE_2 = 0.0002` (paper, D-sarl-lr-2nd)                             | plan D.9                                  | queued |
| D8→D.9      | Adam `betas=(0.95, 0.95)` (paper, D-sarl-adam-beta1/2)                    | plan D.9                                  | queued |
| D8→D.9      | `SCHEDULER_PERIOD = 1` (paper, D-sarl-sched-step)                         | plan D.9                                  | queued |
| D8→D.12     | `num_frames = 500_000` ou `1_000_000` (paper, D-sarl-num-frames — decision pending) | plan D.12                           | queued |

## Cross-reference deviations.md (Phase B + Phase C)

| ID existant                  | Status D.8 review                                                     |
|:-----------------------------|:----------------------------------------------------------------------|
| D-sarl-target-update (🆘)    | **Port = paper (1000)**, student diverge (100). Port ✅ resolved implicitly |
| D-sarl-lr-2nd (🆘+❌)        | Confirmé L81. Queued D.9                                              |
| D-sarl-adam-beta1/2 (🆘+❌) | Confirmé L207, 213 (Adam sans betas, default). Queued D.9            |
| D-sarl-sched-step (🆘+❌)   | Confirmé L83 (1000). Queued D.9                                       |
| D-sarl-sched-gamma (⚠️)     | Info : 0.999 paper silent. Keep.                                      |
| D-sarl-num-frames (🆘)       | Confirmé L128 (5M). Queued D.12                                       |
| D-sarl-alpha-ema (🆘🆘)     | D.2 resolved (config 45). **D8-F3 : aligner dataclass default aussi** |
| D-sarl-gamma (✅)            | D.7 resolved.                                                         |
| D-sarl-recon-bias (✅)       | D.7 resolved.                                                         |
| D-sarl-cascade-noop (⚠️)    | D.4 resolved.                                                         |
| D-sarl-backward-order (⚠️)  | Trainer scope (D.6 reviewed).                                         |
| D-sarl-dropout-position (🆘+⚠️) | Model scope (D.3 reviewed, D.4 handled Option A).                |
| D-sarl-bce-shape (⚠️)       | Trainer scope (D.6 reviewed).                                         |
| D-sarl-setting-7 (❌)        | hors scope SARL, Phase E port ACB                                     |
| D-sarl-seeds (❌)            | matrix scope, B.13 done                                               |

### 2 nouvelles deviations identifiées en D.8

- **D-sarl-batch-size** (🚨 new) : `BATCH_SIZE = 128` vs paper/student 32. Queued D.9.
- **D-sarl-step-size-1** (🚨 new) : `STEP_SIZE_1 = 0.0003` vs paper 0.00025 ~20% typo. Queued D.9.

**Piste D.9 scope update** : 5 fixes au lieu des 3 initialement prévus (lr_second_order +
adam_betas + sched_step) → +batch + step_size_1 + alpha default.

## Résumé — `sarl/training_loop.py`

- ✅ **Architecture student parity** : `_build_networks`, `_build_optimizers`, `run_training`
  main loop, `_persist_outputs`, `setting_to_config` tous bit-exact avec student structure.
- ✅ **D.4 + D.7 integrations** (cascade warning, b_recon propagation via state_dict, gamma
  plumbing) cohérentes.
- ✅ **10 module constants** audités ; 2 match paper exact (REPLAY_*, TARGET_UPDATE_FREQ),
  2 match partial (TRAINING_FREQ, MIN_SQUARED_GRAD), 6 diverge paper (via D-sarl-*).
- 🚨 **2 nouvelles deviations** : D-sarl-batch-size (port 128 vs paper 32), D-sarl-step-size-1
  (port 0.0003 vs paper 0.00025). Queued D.9.
- ⚠️ **D8-F3** : `alpha` dataclass default 1.0 obsolete post-D.2, aligner sur 45. Queued D.9.
- **0 bug logique bloquant.** Le training loop est structurellement solide ; les divergences
  sont toutes des valeurs config paper-vs-port.

**D.8 clôturée. 3 fixes direct pour D.9 (D8-fix-1/2/3) + 4 fixes queued D.9/D.12 déjà tracés. 0 code touché.**
