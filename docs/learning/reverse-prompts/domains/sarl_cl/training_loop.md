# Reverse-prompt — `src/maps/domains/sarl_cl/training_loop.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl_cl/training_loop.py`
**Taille :** ~1040 lignes.
**Paper :** §4, Table 11 CL.

---

## 1. Quelle aurait été la spec ?

> Écris le training orchestrator pour SARL+CL. Structurellement mirror
> `domains/sarl/training_loop.py` mais ajoute la CL machinery.
>
> **Différences vs SARL standard** :
>
> 1. **Teacher networks** (optional) :
>    - `teacher_first_net, teacher_second_net` (frozen)
>    - Drivent les `distillation` + `feature` losses dans
>      `sarl_cl_update_step`
>    - Chargés via `load_partial_state_dict(prev_checkpoint)`
>
> 2. **AdaptiveQNetwork** backbone (optional) :
>    - Si `cfg.adaptive_backbone=True`, utilise `AdaptiveQNetwork` au
>      lieu de `SarlCLQNetwork`
>    - Handles variable in_channels via channel padding
>
> 3. **`load_partial_state_dict`** helper :
>    - Charge un checkpoint dont les shapes ne matchent pas exactement
>    - Skip les params dont les shapes diffèrent (logs warning)
>    - Permet transfer Breakout (4 ch) → Seaquest (10 ch) via Adaptive
>
> 4. **Per-network `DynamicLossWeighter`** :
>    - 1 instance pour FO, 1 pour SO
>    - Maintained across le whole curriculum (running max persistant)
>
> **Paper constants** :
> - `TARGET_NETWORK_UPDATE_FREQ = 500` (CL spécifique, vs SARL standard
>   1000)
> - Tous les autres identical à SARL paper-aligned (BATCH_SIZE=128,
>   STEP_SIZE_1=0.0003, STEP_SIZE_2=0.0002, ADAM_BETAS=(0.95, 0.95),
>   etc.)
>
> **`SarlCLTrainingConfig` dataclass** : tous les knobs SARL + 6
> CL-specific :
> - `curriculum: bool` — enable 3-term CL loss
> - `adaptive_backbone: bool`
> - `max_input_channels: int = 10`
> - `teacher_load_path: Path | None`
> - `weight_task=0.3, weight_distillation=0.6, weight_feature=0.1`
>   (default — overridable to (0.4, 0.4, 0.2) Figure 7)
> - `num_frames=100_000` (paper text p.17 "100k per env × 4 envs", **per
>   stage** — chain via run_sarl_cl.py invocations)
>
> **Curriculum order** (paper p.9) :
> Breakout → SpaceInvaders → Seaquest → Freeway.
> Caller-side (chained `run_sarl_cl.py --teacher-load-path X`),
> pas enforced dans le code.

## 2. Contraintes scientifiques

### Per-stage `num_frames=100_000`

Paper §4 dit "100k frames per environment, 4 environments". Total = 400k.
Mais une seule invocation `run_sarl_cl.py` = 1 stage. Chain via
`--teacher-load-path` du stage précédent.

### Curriculum order non-enforced

Code permet n'importe quel order. Convention paper Breakout → SI →
Seaquest → Freeway. À documenter explicitement dans le CLI help.

### `target_update_freq=500` (CL) vs 1000 (SARL)

Paper Table 11 silent ; student CL utilise 500. Plus fréquent → target
suit mieux le policy → CL gain. Kept paper-faithful via student.

## 3. Contraintes d'ingénierie

### `load_partial_state_dict` semantics

```python
def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()
    matched = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return matched, skipped
```

Permet de charger un checkpoint avec params manquants ou mismatched
shape. Critical pour cross-game transfer.

Logs warning sur skipped keys pour transparency.

### Outer loop structurellement faithful

Mirror SARL training_loop : épisode → step → buffer → ε-greedy →
update → target sync → validation. Pas bit-exact (RNG drift).

### Per-network weighters

```python
loss_weighter_fo = DynamicLossWeighter()
loss_weighter_so = DynamicLossWeighter()
```

Séparés car FO et SO ont des magnitudes très différentes (CAE vs BCE).
Une seule weighter pour les 2 normalizerait mal.

### Imports cross-module

```python
from maps.experiments.sarl.training_loop import _check_first_order_loss_kind
from maps.experiments.sarl.data import SarlReplayBuffer, ...
from maps.experiments.sarl.evaluate import ValidationSummary, ...
from maps.experiments.sarl.rollout import epsilon_greedy_action
```

Réutilise massivement le standard SARL (replay buffer, rollout,
evaluate, fail-fast SimCLR guard). Spécifique CL = trainer +
loss_weighting + model.

## 4. Deviations / dettes présentes

### D-cl-weights (Phase F.4 reversal)

Default weights (0.4, 0.4, 0.2) per Figure 7. Override `-o cl.weight_task=0.3
-o cl.weight_distillation=0.6 -o cl.weight_feature=0.1` pour Table 11
literal.

### D-sarl_cl-max-channels (D.20 resolved)

`max_input_channels=10` paper Table 11.

### D-sarl_cl-num-frames (D.20 resolved)

100_000 per-stage (vs SARL 500_000). Chain 4× via curriculum.

### D-sarl_cl-target-update

500 paper-faithful via student CL.

### D-sarl_cl-curriculum-order (D.19 resolved)

Caller-side controlled. Pas enforced dans le code. Documented dans
docstring `run_sarl_cl.py`.

### Pas de DETTE active

Module gros mais bien structuré. Réutilise SARL standard.

## 5. Questions ouvertes

- **Q1 :** `load_partial_state_dict` skip silently les params mismatched.
  C'est bien, mais comment savoir si TROP de skip = quelque chose qui
  cloche ? Test : threshold (skip >50% des params → raise) ?

- **Q2 :** Per-stage 100k frames assez ? Paper rapporté que oui sur
  Figure 7. Mais peut-être que d'autres game orderings nécessitent
  plus.

- **Q3 :** Le `DynamicLossWeighter` persist across curriculum. Donc
  le running max du first game influence les normalizations sur game
  4. Should we reset between stages ? Trade-off.

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Réutilisation SARL standard (replay, rollout, evaluate, guard)
- `SarlCLTrainingConfig` dataclass
- Per-network DynamicLossWeighter
- `load_partial_state_dict` helper
- `target_update_freq=500` CL-specific
- `num_frames=100_000` per-stage

### À améliorer
- **Threshold skip check** dans load_partial_state_dict
- **Reset weighters between stages** option
- **Curriculum order validation** : si le caller passe les games
  dans un order non-paper, emit warning

### Tests à écrire
- Single-task run (no teacher) → equivalent SARL standard
- Multi-task curriculum : load checkpoint, continue training
- AdaptiveQNetwork : Breakout 4-ch → Seaquest 10-ch transfer
- `load_partial_state_dict` skip mismatched, keep matched
- Per-network weighters maintained across stages

### Opportunités perf
- Curriculum chains naturally parallelizable per-seed mais sequential
  per-game. Pas grand-chose à optim.
- Teacher forward no_grad — déjà fait.

### Connexion
- Hérite de : `sarl.training_loop` (helpers), `sarl.data`,
  `sarl.evaluate`, `sarl.rollout`
- Utilise : `sarl_cl.{model, loss_weighting, trainer}`
- Appelé par : `domains/sarl_cl/cli.py`
