# Reverse-prompt — `src/maps/domains/sarl_cl/trainer.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl_cl/trainer.py`
**Taille :** ~390 lignes.
**Paper :** §4.
**Référence externe :** Kirkpatrick et al. (2017) EWC.

---

## 1. Quelle aurait été la spec ?

> Écris une **pure function** `sarl_cl_update_step(...)` qui ajoute
> les **3 termes CL** au update DQN standard :
>
> - **`task`** : loss tâche courante (CAE Huber pour FO, BCE-with-logits
>   pour SO)
> - **`distillation`** : `weight_regularization(student, teacher)` —
>   L2 anchor (PAS Hinton KL)
> - **`feature`** : MSE between student et teacher activations
>   intermédiaires (h1 pour FO, comparison_out pour SO)
>
> Port `SARL_CL/examples_cl/maps.py:610-829`.
>
> **Args supplémentaires vs SARL trainer** :
> - `teacher_first_net, teacher_second_net` (frozen)
> - `loss_weighter, loss_weighter_second` (`DynamicLossWeighter`
>   instances)
> - `mixing` (`LossMixingWeights` dataclass — task=0.4, distill=0.4,
>   feature=0.2 default Phase F.4)
>
> **Outputs `SarlCLUpdateOutput`** :
> - `loss, loss_second, q_values, wager_logits` (same as SARL)
> - `components_first, components_second` (`SarlCLComponentLosses`
>   dataclasses with task/distill/feat scalars for logging)
>
> **Sequence** :
> 1. Forward 1st-order + 2nd-order (avec cascade)
> 2. Compute `task` loss (CAE / BCE)
> 3. If teacher_first_net is not None :
>    - Forward teacher (no_grad)
>    - `distill = weight_regularization(student, teacher_first)`
>    - `feature = F.mse_loss(student_h1, teacher_h1)`
> 4. `losses = {"task": task, "distillation": distill, "feature":
>    feature}` then `weighter.update(losses); weighted =
>    weighter.weight_losses(losses)`
> 5. `loss = mixing.task · weighted["task"] + mixing.distillation ·
>    weighted["distillation"] + mixing.feature · weighted["feature"]`
> 6. **Two-loss gradient pattern** (same as SARL trainer) :
>    - `loss_second.backward(retain_graph=True); optim2.step()`
>    - `loss.backward(); optim.step()`
> 7. Schedulers step

## 2. Contraintes scientifiques

### Two-backward pattern (load-bearing)

Identique à SARL standard. Documenté en `domains/sarl/trainer.md` Q1.
Sequence `loss_2.backward → optim2.step → loss.backward → optim.step`
crucial pour parity bit-exact.

### `distillation` = L2 anchor, pas KL distillation

Paper appelle ça "distillation" mais c'est `weight_regularization`
(L2 sur params student vs teacher). Kirkpatrick 2017 EWC sans Fisher
weighting.

Le `core/losses.distillation_loss` (Hinton KL) **n'est jamais appelé**
en prod. Le port garde DistillationLoss définie pour fidélité student
(qui le fait pareil).

### Feature MSE

`F.mse_loss(student_h1, teacher_h1)` où h1 est l'hidden 128-d (FO) ou
comparison_out 1024-d (SO). Force le student à garder les "features"
du teacher pour préserver les compétences passées.

C'est conceptuellement entre EWC (anchor sur params) et feature
distillation (anchor sur activations). MAPS combine les 2.

### Paper Table 11 CL hyperparams

- `weight_task=0.3, weight_distill=0.6, weight_feature=0.1` (Table 11)
- vs paper text p.17 "optimal" (0.4, 0.4, 0.2) — Figure 7 canon
- vs student code (1.0, 1.0, 1.0)

Sprint-08 D.20 puis Phase F.4 reversal → port default (0.4, 0.4, 0.2).

## 3. Contraintes d'ingénierie

### Teacher must be frozen

`teacher.requires_grad_(False)` MUST be done par le caller. Sinon
`weight_regularization` flow gradient dans teacher → teacher drift →
silent correctness bug.

Documenté dans `core/losses.weight_regularization` docstring.

### `DynamicLossWeighter` per-network

Une instance pour 1st-order, une pour 2nd-order. Permet à chaque
network d'avoir ses propres running maxes (FO et SO ont des magnitudes
différentes).

### `LossMixingWeights` dataclass vs scalar args

Pourquoi un dataclass plutôt que 3 scalar args ? **Clarity**. Un
appel `sarl_cl_update_step(..., mixing=LossMixingWeights(task=0.4,
...))` est plus lisible que `..., weight_task=0.4, weight_distill=0.4,
weight_feature=0.2`.

### `SarlCLUpdateOutput.components_*` for logging

Permet d'introspecter quel term domine la loss combinée. Critical
pour debug curriculum runs où parfois un term explose.

## 4. Deviations / dettes présentes

### D-cl-weights (Phase F.4 reversal)

Voir §2. Default (0.4, 0.4, 0.2) per Figure 7.

### D-sarl_cl-target-update

`TARGET_NETWORK_UPDATE_FREQ = 500` (CL) vs 1000 (SARL standard).
Student CL utilise 500 (paper Table 11 silent, kept).

### D-sarl_cl-backward-order

Same as SARL — paper-faithful, ne pas swap.

### DETTE-3 cross-ref

`core.losses.distillation_loss` est dead code, `weight_regularization`
est le vrai "distillation" en CL. Voir `core/losses.md`.

### Pas de DETTE active sur ce module

## 5. Questions ouvertes

- **Q1 :** Pourquoi paper Table 11 vs text p.17 vs student divergent
  sur les weights ? Probable que les 3 ont été essayés à différents
  moments du paper writing. Figure 7 = canonical (référence à ce qui
  a été rapporté).

- **Q2 :** L'L2 weight anchor vs feature MSE vs combined — quelle est
  la contribution de chacun ? Ablation D.20a faite ?

- **Q3 :** `weight_regularization` flow gradient bidirectionnel si
  teacher pas frozen. Quel impact si caller oublie `requires_grad_(False)` ?

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Pure function (no globals)
- `LossMixingWeights` dataclass
- `SarlCLComponentLosses` for logging
- Two-backward sequence paper-faithful
- Per-network `DynamicLossWeighter`
- Default weights (0.4, 0.4, 0.2) Phase F.4

### À améliorer
- **Assert teacher frozen** au début (param `teacher.requires_grad`
  check). Coût trivial, évite silent bug.
- **Type hints `nn.Module`** pour networks
- **Documenter clairement** le distill = L2 anchor, pas Hinton KL

### Tests à écrire
- Bit-exact parity vs reference
- No teacher → equivalent to SARL trainer (with task only)
- With teacher : 3 components computed
- Teacher not frozen raise (if assert added)
- Mixing weights respected

### Connexion
- Utilise : `core.losses.weight_regularization`, `domains.sarl.losses.cae_loss`
- Utilise : `domains.sarl.data.Transition`
- Utilise : `domains.sarl_cl.loss_weighting.DynamicLossWeighter`
- Appelé par : `domains.sarl_cl.training_loop.run_training_cl`
