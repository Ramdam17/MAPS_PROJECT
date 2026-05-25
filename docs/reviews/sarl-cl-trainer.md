# Component review — `src/maps/experiments/sarl_cl/trainer.py`

**Review sub-phase :** Sprint-08 D.18.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl_cl/trainer.py` (374 L, 3 dataclasses + 1 pure function).
**Paper sources :** §2.3 (CL) + §4, eq.15-17 (normalization), eq.17-18 (mixing), Table 11 rows 20-23
(CL-specific weights).
**Student sources :** `external/paper_reference/sarl_cl_maps.py:610-829` (`train()` — CL branch).

**Callers du port :**
- `sarl_cl/training_loop.py:402` (via `sarl_cl_update_step(...)`).

**DoD D.18** : doc créé, 2 branches auditées (non-CL + CL), cross-ref D.6 SARL + D.16 weighter + D.17 model, 0 code touché.

---

## (a) Architecture — 2 branches unifiées

`sarl_cl_update_step` se comporte différemment selon `cl_enabled = teacher_first_net is not None` :

- **`cl_enabled=False`** : degenerate path. Équivalent à `sarl.sarl_update_step` sauf sur :
  - `cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)` utilise `W = state_dict()['fc_hidden.weight']` **du CL variant** (explicit decoder) — même interface, même Jacobian math.
  - Comparison `flat_input - Output` vs SARL `flat_input - ReLU(W^T·h)` (tied). Cascade sur Output 1024-dim (D.17).
- **`cl_enabled=True`** : 3-term loss per network.
  - FO : `task = cae_loss(...)`, `distillation = weight_regularization(policy, teacher_first)`, `feature = F.mse_loss(h1, h1_teacher)`.
  - SO : `task = BCE-with-logits(wager, targets_wagering)`, `distillation = weight_regularization(second, teacher_second)`, `feature = F.mse_loss(comparison_out, comparison_out_teacher)`.
  - Both normalized via `loss_weighter.update(...) + loss_weighter.weight_losses(...)` (paper eq.15-17), then mixed via `mixing.task * weighted_task + mixing.distillation * weighted_reg + mixing.feature * weighted_feature`.

### (a1) 🚨 D18-F1 — docstring L21-23 mensonger sur paper weights

Docstring module L21-23 dit :

> *"These are normalized by a :class:`DynamicLossWeighter` (one per network) via running-max
> division, then mixed with fixed scalar weights [...]. The paper uses (1.0, 1.0, 1.0) —
> captured here via the ``LossMixingWeights`` dataclass."*

**"The paper uses (1.0, 1.0, 1.0)"** est **factuellement faux**. Extraction paper :
- **Paper Table 11** rows 20-22 : task=0.3, reg=0.6, feature=0.1.
- **Paper text p.17** : *"optimal weights (0.4, 0.4, 0.2)"* — contradit Table 11.
- **Student `sarl_cl_maps.py:708-710`** (`WEIGHT1=WEIGHT2=WEIGHT3=1.0`) : (1, 1, 1) — **student diverges paper** sur les 2 sources.

→ **3-way disagreement** (Table 11 ≠ text ≠ student), capturé par `D-cl-weights` (🆘+❌).

Docstring port dit "paper uses (1,1,1)" en confondant **student** avec **paper**. À corriger.

**Piste D18-fix-1** : corriger docstring L21-23 pour documenter la 3-way divergence honnêtement,
pointer `D-cl-weights`. **Not blocking** (docstring only), queued pour C.10-style batch ou D.20.

## (b) Mapping port ↔ student (cl_enabled=True path)

Student `sarl_cl_maps.py:765-795` verbatim (approximatif) :
```python
# FO CL branch
loss_task = CAE_loss(W, target, Q_s_a, h1, lam)
loss_previous_task = compute_weight_regularization(policy_net, teacher_first_net)
feature_loss = f.mse_loss(h1, h1_teacher)
current_losses = {'task': loss_task, 'distillation': loss_previous_task, 'feature': feature_loss}
loss_weighter.update(current_losses)
weighted_losses = loss_weighter.weight_losses(current_losses)
loss = (WEIGHT2 * weighted['distillation'] + WEIGHT1 * weighted['task'] + WEIGHT3 * weighted['feature'])
```

Port L285-302 :
```python
loss_task = cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)
loss_distillation = weight_regularization(policy_net, teacher_first_net)
loss_feature = F.mse_loss(h1, h1_teacher)
raw = {"task": loss_task, "distillation": loss_distillation, "feature": loss_feature}
loss_weighter.update(raw)
weighted = loss_weighter.weight_losses(raw)
loss = mixing.task * weighted["task"] + mixing.distillation * weighted["distillation"] + mixing.feature * weighted["feature"]
```

✅ **Bit-parity student** sur les 7 ops FO (task / reg / feature / weighter update / weighter
weight / fixed mix).

Same pattern for SO branch (port L330-346 vs student L800-830). ✅ Bit-parity.

## (c) 🚨 Backward order — load-bearing (même que SARL D.6)

Port L356-364 :
```python
loss_second.backward(retain_graph=True)   # ① SO grads → policy_net too
optimizer2.step()                           # ② SO weights move
loss.backward()                             # ③ FO grads accumulate (+ ① residuals)
optimizer.step()                            # ④ policy_net moves
scheduler1.step(); scheduler2.step()        # ⑤ LR decay
```

Student `sarl_cl_maps.py:820-830` même ordre. Load-bearing comme SARL D.6 :
- `loss_second` depend on `wager → comparison_out → comparison_1 → policy_net` (via CL decoder
  explicit — note : pas de tied-weight ici, mais `comparison_1 = flat_input - Output` dans
  `SarlCLQNetwork` dépend quand même de `policy_net` forward).
- `optimizer2.step()` ne touche que `second_order_net` → `policy_net.grad` from ① reste intact.
- `loss.backward()` ADD aux grads existants (accumulation standard PyTorch).
- `optimizer.step()` update `policy_net` avec sum ① + ③.

✅ **Port = student** bit-parity sur cross-loss coupling.

Docstring module L27-37 documente l'intention → explicite et clair.

## (d) Non-CL degenerate path (cl_enabled=False)

Port L303-316 :
```python
else:
    loss = cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)

# SO branch (if meta=False)
if not meta:
    if train:
        loss.backward()
        optimizer.step()
        scheduler1.step()
    return SarlCLUpdateOutput(loss=loss, q_values=q_s_a, components_first=None)
```

**Équivalent fonctionnel à `sarl.sarl_update_step`** sur le path non-meta non-CL. Différences :
- CL variant `SarlCLQNetwork` → decoder explicit → `comparison = flat_input - Output` direct vs SARL tied-weight.
- `W = state_dict()['fc_hidden.weight']` → même Jacobian (cae_loss math identique).

→ Permet de run **pure SARL via CL networks** (ex: task-1 of a curriculum avant qu'un teacher existe).

## (e) `target_wager_fn` + `gamma` + `cascade_iterations_*` injection

Mêmes pattern que D.6 post-D.7 :
- `gamma: float = 0.999` keyword-only ✅ (paper-faithful default post-D.7).
- `target_wager_fn`, `mixing`, `cascade_iterations_*` injectés pour testabilité.

✅ Cohérence post-D.7.

## (f) `train=False` — validation-only path

Port L307-311, L356-364 skippent `backward + step + scheduler.step` quand `train=False`.
Utilisé par `training_loop.py` en mode validation (appel dans une boucle eval sans muter poids).

Student `sarl_cl_maps.py:evaluation()` fait équivalent via `train_or_test=False` argument.

✅ **Parity student** sur l'observation-only path.

## (g) `asserts` invariants — fail-fast

Port L203-215 :
```python
if meta:
    assert second_order_net is not None
    assert optimizer2 is not None
    assert scheduler2 is not None

if cl_enabled:
    assert loss_weighter is not None
    if meta:
        assert teacher_second_net is not None
        assert loss_weighter_second is not None
```

**Gain port** : fail-fast au lieu d'erreurs cryptiques (NoneType attribute access). Student pas
d'asserts. Non-breaking.

## Cross-reference deviations.md / reviews existants

| ID existant                 | Status D.18 review                                                 |
|:----------------------------|:-------------------------------------------------------------------|
| D-cl-weights (🆘+❌)       | Confirmé : port utilise `LossMixingWeights(1,1,1)` default, paper Table 11 dit 0.3/0.6/0.1, text dit 0.4/0.4/0.2. Queued D.20. |
| D-sarl_cl-num-frames (❌)  | N/A in trainer.py (config scope). Queued D.20. |
| D-sarl_cl-max-channels (❌) | N/A in trainer.py (model scope). Queued D.20. |
| D-sarl_cl-channel-adapter (✅) | Resolved D.17. |
| D-sarl_cl-lossweight-normalization (⚠️) | Re-confirmed via `loss_weighter.update + weight_losses` usage. |
| D-sarl-backward-order (⚠️) | Same cross-loss coupling pattern ici. Load-bearing docstring preserved. |
| DETTE-1 (triplé) | `SarlCLSecondOrderNetwork` usage ici confirme DETTE-1 D.17 extension. |
| DETTE-3 `distillation_loss` dead | Port utilise `weight_regularization` (L2 anchor), PAS `distillation_loss` (KL). Cohérent DETTE-3 docstring claim. |

### Aucune nouvelle deviation surfacée en D.18.

## Fixes identifiées D.18

| ID        | Fix                                                                       | Scope                                  | Effort |
|:----------|---------------------------------------------------------------------------|----------------------------------------|:------:|
| D18-fix-1 | Docstring L21-23 module — corriger "paper uses (1,1,1)" (false) → documenter 3-way divergence (Table 11=0.3/0.6/0.1, text=0.4/0.4/0.2, student=1/1/1) | `sarl_cl/trainer.py:21-23` | 5 min |
| D18→D.20  | Align `LossMixingWeights` default + yaml vers paper Table 11 (0.3/0.6/0.1) | queued D.20                            | queued |
| D18-fix-2 | (info, skip) docstring CAE_LAMBDA rationale                                | —                                      | skip |

## Résumé — `sarl_cl/trainer.py`

- ✅ **Architecture 2-branches** (non-CL degenerate + CL 3-term) bit-parity student `train()`.
- ✅ **FO + SO 3-term composition** bit-parity 7 ops per network.
- ✅ **Backward order load-bearing** même pattern que SARL D.6 — cross-loss coupling preserved.
- ✅ **`train=False` validation path** — skip backward/step, parity student `train_or_test=False`.
- ✅ **Fail-fast asserts** pour meta + cl_enabled invariants (gain port, non-breaking).
- ✅ **`target_wager_fn`/`gamma`/`mixing` injection** — testable, post-D.7 coherent.
- 🚨 **D18-F1** : docstring module L21-23 mensonger sur paper weights (claim "(1,1,1)" = student
  value, pas paper). Fix trivial. D18-fix-1.
- ⚠️ **D-cl-weights** re-confirmé `mixing(1,1,1)` default. Real fix queued D.20.
- **0 nouvelle deviation, 1 docstring fix, 1 queued D.20.**

**D.18 clôturée. 1 fix docstring trivial + 1 queued D.20 (weights). 0 code touché.**
