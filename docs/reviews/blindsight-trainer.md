# Component review — `src/maps/experiments/blindsight/trainer.py`

**Review sub-phase :** Sprint-08 D.23.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/blindsight/trainer.py` (452 L).
**Paper sources :** §2.2 Blindsight, Table 9 Blindsight hyperparams, paper text p.13 "Results
Blindsight".
**Student sources :** `external/paper_reference/blindsight_tmlr.py` (`prepare_pre_training` L460-,
`pre_train` L474-, `testing` L806-).

**Callers du port :** `scripts/run_blindsight.py` (Sprint-06 CLI driver).

**DoD D.23** : doc créé, pre_train + evaluate audités, **RG-002 diagnostic consolidé**, 0 code touché.
Fixes queued D.25 pour RG-002.

---

## (a) Module structure

- `BlindsightSetting` dataclass (L77) — setting flags (meta on/off, cascade on/off).
- `_build_optimizer` (L101) — Adam / Adamax factory.
- `_condition_params_from_env` (L108) — cfg-driven params par condition.
- `BlindsightTrainer` class (L117-452) :
  - `__init__` (L134) — cfg load + device + cascade params.
  - `_load_env_cfg` (L163) — merge env YAML (superthreshold / subthreshold / low_vision condition params).
  - `build` (L179) — networks + optimizers + schedulers. **Post-D.22b guard-rail présent L187-189 ✅**.
  - `pre_train` (L229) — cœur training loop (CAE + wagering BCE).
  - `evaluate` (L354) — cœur eval : discrimination_accuracy + wager_accuracy.

## (b) `build()` — post-D.22b guard-rail ✅

Post-D.22b (2026-04-20) : le trainer appelle `_check_first_order_loss_kind` au début de `build()` via import interne :
```python
from maps.experiments.sarl.training_loop import _check_first_order_loss_kind
_check_first_order_loss_kind(str(self.cfg.get("first_order_loss", {}).get("kind", "cae")))
```

✅ **Garde-fou validé** : un run avec `kind='simclr'` raise `NotImplementedError` au build, avant tout compute.

## (c) `pre_train()` loop — bit-parity student

### Structure L281-350

1. Generate batch (patterns + stim_present + order_2_target).
2. First-order cascade forward : `(h1, h2)` post 50 iters.
3. `batch.patterns.requires_grad_(True)` + `h2.requires_grad_(True)` — **reference quirk** (student L569-572 fait pareil, préserve le graph).
4. `optim_1.zero_grad()`.
5. If `setting.second_order` :
   - Second-order cascade forward → `wager`.
   - `loss_2 = wagering_bce_loss(wager.squeeze(-1), target, reduction="sum")`.
   - `loss_2.backward(retain_graph=True)` + `optim_2.step()` + `sched_2.step()`.
6. Else : throwaway 2nd-order forward (no-grad) — **preserve RNG consumption pour parity student**.
7. First-order CAE loss : `cae_loss(W=self.first_order.fc1.weight, x=batch.stim_present.view(-1, num_units), recons_x=h2, h=h1, lam=lam, recon="bce_sum")`.
8. `loss_1.backward(retain_graph=True)` + `optim_1.step()` + `sched_1.step()`.

### (c1) Backward order load-bearing

Identique pattern SARL D.6 + SARL+CL D.18 :
- `loss_2.backward(retain_graph=True)` d'abord → gradients via `comparison` → aussi `first_order.fc1`.
- `optim_2.step()` mais `first_order.grad` intact.
- `loss_1.backward(retain_graph=True)` → ADD to `first_order.grad`.
- `optim_1.step()` → update first_order avec ①+③.

✅ **Bit-parity student** (L573-581 `blindsight_tmlr.py`).

### (c2) `cae_loss` input — `x=batch.stim_present` vs `recons_x=h2`

Port L341 : `x=batch.stim_present.view(-1, num_units)` comme TARGET reconstruction (clean stimulus).
Port L342 : `recons_x=h2` comme PRED (decoder output).

Student `blindsight_tmlr.py:595` call : `criterion_1(W, patterns_tensor, output_first_order, hidden_representation, lam)` where `patterns_tensor = stim_present_tensor` (stim_present = clean stimulus). ✅ match.

⚠️ **Subtle** : le decoder sort `h2` qui essaie de reproduire `stim_present` à partir de `patterns` (patterns = stim_present + noise). C'est un **denoising autoencoder**, pas un simple autoencoder. Paper-faithful.

### (c3) `loss_1.backward(retain_graph=True)` — retain pour rien ?

Port L347. Mais après L348 `optim_1.step()` il n'y a plus de backward. `retain_graph=True` inutile ici.

**Student L603** fait pareil → parity preserved.

**Piste D23-fix-1** (cosmetic, low prio) : le `retain_graph=True` sur le 2nd backward est redundant. Student parity, pas un bug.

### (c4) `requires_grad_(True)` L303-305 après le forward

Port L303 `batch.patterns.requires_grad_(True)` + L305 `h2.requires_grad_(True)`. **Mutation in-place** post forward.

**Student L567-568** fait pareil. Parity preserved.

**Utilité** : en principe, après un forward sans `.no_grad()`, les tensors tracked ont déjà `requires_grad=True` via leur autograd provenance. Student is paranoid / defensive.

## (d) `evaluate()` — 🚨 RG-002 DIAGNOSTIC FOCUS

### Structure (3 conditions × 200 trials each)
Pour chaque cond dans {superthreshold, subthreshold, low_vision} :
1. Genère 200 trials (100 noise-only + 100 stim-present).
2. Forward cascade 50 iters → `h2`.
3. **Metric 1 `discrimination_accuracy`** : `h2[stim:].argmax == batch.patterns[stim:].argmax` (sur 100 stim-present seulement).
4. If meta : forward second_order 50 iters → `wager`.
5. **Metric 2 `wager_accuracy`** : `(high_w > threshold) == (tgt > threshold)` sur 100 stim-present.

### (d1) Discrimination metric — bit-parity student

Port L428-430 :
```python
pred_idx = h2[delta:].argmax(dim=1)
true_idx = batch.patterns[delta:].argmax(dim=1)
discrimination_acc = float((pred_idx == true_idx).float().mean().item())
```

Student `testing()` L870 :
```python
discrimination_performance = round(
    (output_first_order[delta:].argmax(dim=1) == input_data[delta:].argmax(dim=1))
    .to(float).mean().item(), 2
)
```

✅ **Math bit-exact**. Seule diff : student round à 2 décimales (cosmétique).

### (d2) Wager accuracy — bit-parity en formule, edge case trivial

Port L442-447 :
```python
high_w = wager[delta:, 0].cpu().numpy()
tgt = batch.order_2_target[delta:, 0].detach().cpu().numpy()
pred_bin = (high_w > threshold).astype(int)
tgt_bin = (tgt > threshold).astype(int)
wager_acc = float((pred_bin == tgt_bin).mean())
```

Student L897-903 :
```python
TP = np.sum((predicted_np >  threshold) & (targets_2_np > threshold))
TN = np.sum((predicted_np <  threshold) & (targets_2_np < threshold))
FP = np.sum((predicted_np >  threshold) & (targets_2_np <  threshold))
FN = np.sum((predicted_np <  threshold) & (targets_2_np >  threshold))
precision_h, recall_h, f1_score_h, accuracy_h = compute_metrics(TP, TN, FP, FN)
# accuracy_h = (TP+TN)/(TP+TN+FP+FN)
```

**Mathématiquement équivalent** : `(pred_bin == tgt_bin).mean() == (TP+TN)/(TP+TN+FP+FN)`.

⚠️ **Edge case mineur** : student utilise strict `<` et `>` — si une valeur `== threshold` exactement, elle est **exclue** du comptage TP+TN+FP+FN. Port la compte comme correcte (pred_bin = tgt_bin = 0 tous les 2). Sur 100 trials, edge cases avec wager==threshold sont **rarissimes** (continuous float). Contribution à l'écart z=0.40 vs z=9.01 : **négligeable**.

### (d3) 🚨 RG-002 consolidation — 2 hypothèses

#### H1 : `hidden_dim=100` vs paper 60 (structural, C11-F2 confirmé)
- Port config `blindsight.yaml:10 hidden_dim=100`.
- Paper Table 9 row 6: `Hidden size = 60`.
- Effect : paper a bottleneck strict `100→60→100` = force compression. Port `100→100→100` pas de bottleneck → 1st-order peut sur-fit sans apprendre representation compressée → discrimination plus faible car pas de "squeeze" invariance.
- **Hypothèse forte** pour expliquer l'écart z.
- Fix queued D.25 : config yaml change `hidden_dim: 60`.

#### H2 : Paper "detection accuracy" définition (ambiguous)
Paper reports **"detection accuracy z=9.01"** en Table 5. Il existe plusieurs définitions candidates :
- **Discrimination accuracy** moyenne sur 3 conditions ? Port calcule ça (3 conditions, moyenne).
- **Wager accuracy** moyenne ? Port calcule ça aussi.
- **Une combinaison** (weighted avg, AUC, ou max-across-conditions) ?
- **Paper silent** sur l'agrégation précise.

Port fait `{cond: {discrimination_accuracy, wager_accuracy}}` — caller Sprint-06 report aggregates how? À vérifier en D.25 scope.

**Piste diagnostic D.25** :
1. Aligner `hidden_dim=60` (H1 fix).
2. Re-run 10 seeds smoke (local CPU) → observer z-score new.
3. Si H1 fix suffit → RG-002 resolved.
4. Si écart persiste → investigate H2 : tester plusieurs agrégations (mean subthreshold only vs mean all vs max) et voir laquelle matche paper.

### (d4) Threshold logic

Port L395 `thresholds = eval_cfg.wager_thresholds`. Student L816 : 0.5 pour superthreshold / subthreshold, 0.15 low-vision. Cross-ref paper silent, student arbitrary. Port config-driven → override possible.

⚠️ **Low-vision threshold 0.15** (vs 0.5 standard) abaisse la barre : plus de bets "high" → plus d'accuracy mais artificiellement. Pas un bug mais un **hyperparameter** arbitraire student. À documenter paper-vs-student si on veut être exhaustif.

## (e) Cross-ref deviations.md

| ID existant             | Status D.23 review                                                           |
|:------------------------|:------------------------------------------------------------------------------|
| **RG-002** (Blindsight z gap) | Re-confirmed. H1 (hidden_dim=100 vs 60) = structural, queued D.25. H2 (metric agg) = diagnostic pending post-H1-fix. |
| D-001 (wager 2-unit)    | Blindsight uses `n_wager_units=1` (config maps.yaml:23) — student parity. Code reviewed C.4/C.6, post-C.6 fix supports `n_wager_units=2` raw logits si config flip. |
| D-002 (SimCLR vs CAE)   | Resolved D.22b (keep CAE) ; guard-rail en build(). |
| D-004 (AGL chunked sigmoid) | N/A Blindsight (uses global sigmoid decoder, not chunked). |

## Fixes identifiées D.23

| ID        | Fix                                                                       | Scope                                  | Effort |
|:----------|---------------------------------------------------------------------------|----------------------------------------|:------:|
| D23→D.25  | Align `blindsight.yaml:10 hidden_dim: 100 → 60` (H1 RG-002)               | queued D.25                            | queued |
| D23→D.25  | Diagnostic H2 aggregation post-H1-fix (compare paper "detection accuracy" definition) | queued D.25                  | queued |
| D23-fix-1 | (cosmetic, skip) `retain_graph=True` redundant on 2nd backward L347       | `blindsight/trainer.py:347`            | skip   |

## Résumé — `blindsight/trainer.py`

- ✅ **Architecture + build()** cohérents, D.22b guard-rail en place.
- ✅ **`pre_train()` loop** bit-parity student : 8 steps, backward order load-bearing, denoising-CAE + wagering-BCE.
- ✅ **`evaluate()` math** : discrimination + wager accuracy bit-parity student (math-equivalent, edge case trivial).
- 🚨 **RG-002 H1 (structural)** : `hidden_dim=100` vs paper `60` — **hypothèse principale** pour z-gap 0.40 vs 9.01. Fix D.25.
- ⚠️ **RG-002 H2 (diagnostic)** : paper "detection accuracy" agrégation ambiguë. Post-H1-fix en D.25, compare 3 agrégations potentielles.
- ⚠️ **Low-vision threshold 0.15 student-arbitrary** (paper silent). Documentation only.
- **0 nouvelle deviation, 2 fixes queued D.25 (H1 + H2 diagnostic), 1 cosmetic skip.**

**D.23 clôturée. 0 code touché. Next : D.24 (blindsight/data.py review) puis D.25 (RG-002 fix).**
