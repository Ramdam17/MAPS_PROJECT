# Component review — `src/maps/experiments/blindsight/data.py`

**Review sub-phase :** Sprint-08 D.24.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/blindsight/data.py` (161 L, 1 enum + 2 dataclasses +
1 function).
**Paper sources :** paper §A.1 Blindsight (Weiskrantz 1986 / Pasquali & Cleeremans 2010 regimes).
**Student sources :** `external/paper_reference/blindsight_tmlr.py:generate_patterns` L259-328.
**Env config :** `config/env/blindsight.yaml` — 3 conditions paramétrées.

**Callers du port :**
- `src/maps/experiments/blindsight/trainer.py:pre_train` L286, `evaluate` L404.

**DoD D.24** : doc créé, math bit-parity student confirmée, 0 code touché.

---

## (a) Architecture

- `StimulusCondition` IntEnum (L31) : `SUPERTHRESHOLD=0 / SUBTHRESHOLD=1 / LOW_VISION=2`.
  Ints match student `condition` arg, utilisable en enum ou int brut.
- `ConditionParams` dataclass frozen (L43) : 3 floats `random_limit, baseline, multiplier`.
- `TrainingBatch` dataclass (L56) : `patterns, stim_present, order_2_target` tenseurs.
- `generate_patterns(params, patterns_number, num_units, factor, device, rng)` (L77) — cœur.

### Architectural gain over student

Student hardcode 3 branches `if condition == 0/1/2:` inline. Port factorize via
`ConditionParams` loaded from yaml → **config-driven**, abilité à paramétrer des conditions
au-delà des 3 paper sans toucher le code.

`config/env/blindsight.yaml:conditions` :
- `superthreshold : (random_limit=0.0, baseline=0.0, multiplier=1.0)` — student L278-281 match.
- `subthreshold : (random_limit=0.02, baseline=${train.noise_level}, multiplier=1.0)` — L283-286 match.
- `low_vision : (random_limit=0.02, baseline=${train.noise_level}, multiplier=0.3)` — L288-291 match.

✅ **Bit-parity conditions** via config.

## (b) `generate_patterns` math — bit-parity student

Port L114-146 vs student L269-321 :

| Step                                 | Student                                           | Port                                         | Match |
|:-------------------------------------|:--------------------------------------------------|:---------------------------------------------|:-----:|
| N = patterns_number × factor         | `patterns_number = patterns_number * factor`     | `n = int(patterns_number * factor)`          | ✅    |
| Noise-only (1st half)                | `pattern = multiplier * np.random.uniform(0, random_limit, num_units) + baseline` | idem L125-128 | ✅ |
| Noise-only `stim_present`            | `np.zeros(num_units)`                              | idem L130                                    | ✅    |
| Noise-only `order_2_pr`              | `[0.0, 1.0]` (low wager)                           | idem L131                                    | ✅    |
| Stim-present — stim_idx              | `random.randint(0, num_units - 1)` (Python stdlib) | idem L134                                    | ✅    |
| Stim-present — pattern base          | `np.random.uniform(0.0, random_limit, num_units) + baseline` (**no multiplier on base!**) | idem L135 | ✅ |
| Stim-present — stim spike            | `pattern[stim_idx] = np.random.uniform(0.0, 1.0) * multiplier` | idem L136 | ✅    |
| Detection threshold                  | `pattern[stim_idx] >= multiplier / 2`              | idem L140                                    | ✅    |
| `order_2_pr` labeling                | `[1.0, 0.0]` if above, `[0.0, 1.0]` else           | idem L141, L144                              | ✅    |
| `stim_present[stim_idx] = 1.0`       | only if above threshold                            | idem L142                                    | ✅    |

✅ **9 steps math bit-exact**. Aucune divergence.

### (b1) 🚨 Subtlety — noise-only uses `multiplier` on noise, stim-present doesn't

Student L299 (noise half) : `multiplier * np.random.uniform(0, random_limit, num_units) + baseline`.
Student L307 (stim half) : `np.random.uniform(0, random_limit, num_units) + baseline` — **no multiplier**.

Port L125-128 + L135 reproduct this asymmetry exactly.

**Implication** : for `low_vision` (multiplier=0.3), noise baseline in the **noise-only** half is
scaled by 0.3 (so 0.3×0.02 = 0.006 max), while noise in the **stim-present** half is 0.02 max (×~3.3).
**Asymmetric dataset** → noise-only and stim-present trials have different noise scales in the
low_vision condition. Paper-faithful via student, but mathematically odd.

**Not a bug** — student code does this, preserve parity. Documented.

### (b2) RNG parity

- `np.random.uniform` (legacy NumPy global singleton) — seedé via `set_all_seeds` ✅.
- `random.randint` (Python stdlib global) — seedé via `set_all_seeds` ✅.
- Port L116 `np_rng = rng or np.random` — falls back to global for parity.
- Port L112 imports `random` locally (local import, comment "matches reference behavior") — Python global singleton.

✅ **RNG streams paper-faithful**.

**Piste D24-note** (skip) : Port param `rng: np.random.Generator | None` allows test isolation
(new-API Generator). Unused by production (falls back to legacy global). Gain port safe.

## (c) `requires_grad_(True)` on all 3 output tensors (L153-155)

Port commentaire L151-152 : *"The reference calls `.requires_grad_(True)` on these; replicate
for parity, even though only `patterns` actually feeds into an autograd path."*

Student L323-326 fait pareil. Parity preserved despite only `patterns` being meaningful for
grad flow.

✅ **Defensive paper quirk preserved**.

## (d) `stim_absent` tensor — dropped in port

Student `generate_patterns` returns 4 tensors : `patterns, stim_present, stim_absent, order_2`.
Mais `stim_absent = []` (L275) **n'est jamais append'd** dans la boucle → toujours un empty tensor.

Port L147-149 : **drop the dead tensor**. Return signature = `TrainingBatch(patterns, stim_present,
order_2_target)` — 3 tensors, cleaner.

✅ **Port gain** : suppression d'un artefact étudiant mort sans impact math.

## (e) Pre-training condition hardcoded to superthreshold

Per `config/env/blindsight.yaml:pre_training_condition: superthreshold` + student L530. Cross-ref
`trainer.py:_pre_training_params`.

Paper §2.2 confirms pre-training on superthreshold. ✅

## (f) Eval patterns — 200 trials (100 noise + 100 stim)

Port `config/env/blindsight.yaml:eval.patterns_number=200`. Student `testing()` L815 same.
Half noise, half stim-present. ✅

### Wager thresholds
- `superthreshold: 0.5`
- `subthreshold: 0.5`
- **`low_vision: 0.15`** — student L816 uses **arbitrary lower threshold** for low_vision (makes
  sense : stimulus intensity is only 0.3× so threshold should scale). Paper silent on this
  choice. Port paramétrize via yaml — override-able.

⚠️ Low-vision threshold 0.15 = student choice, not paper-prescribed. Documented in trainer D.23
review §(d4). Not a bug.

## (g) RG-002 relevance

**None surfaced D.24**. Data generation is bit-parity student → no hidden RG-002 cause here.
RG-002 root causes remain :
- H1 : `hidden_dim=100` vs paper 60 (structural, D.25 fix).
- H2 : paper "detection accuracy" aggregation (diagnostic pending D.25 post-H1).

## Cross-reference deviations.md

- **Aucune deviation surfacée D.24.**
- `ConditionParams` config match student 3 conditions exactly.
- `generate_patterns` math bit-parity 9 steps.

## Fixes identifiées D.24

| ID        | Fix                                                                     | Scope                           | Effort |
|:----------|-------------------------------------------------------------------------|---------------------------------|:------:|
| D24-note  | Config-driven `ConditionParams` → already a port gain                   | —                               | done  |
| D24-note2 | `rng: Generator` param added for test isolation                         | —                               | done  |
| D24-fix-1 | (debate, skip) asymmetric noise scaling noise-half vs stim-half         | preserve student parity         | skip   |
| D24-fix-2 | (debate, skip) `.requires_grad_(True)` on stim_present / order_2_target | preserve student parity         | skip   |

Aucun fix actionnable — tout bit-parity student, divergences intentionnelles documentées.

## Résumé — `blindsight/data.py`

- ✅ **`generate_patterns` math 9 steps bit-parity student** — aucune divergence.
- ✅ **ConditionParams dataclass** + config yaml = gain port config-driven paper-preserving.
- ✅ **RNG streams** paper-faithful (np.random legacy + Python random — both seedable via
  `set_all_seeds`).
- ✅ **`stim_absent` dead tensor dropped** (port cleanup, math-neutral).
- ⚠️ **Subtle asymmetry** noise-only half scales by multiplier, stim-present half doesn't —
  paper-faithful via student, documented.
- ⚠️ **Low-vision threshold 0.15** (student arbitrary, paper silent) — documented, config-overridable.
- **0 nouvelle deviation, 0 RG-002 cause here.**

**D.24 clôturée. 0 code touché. Next : D.25 (RG-002 fix — hidden_dim 100→60 + metric diagnostic).**
