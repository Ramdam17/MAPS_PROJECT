# Component review — `src/maps/experiments/sarl/trainer.py`

**Review sub-phase :** Sprint-08 D.6.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl/trainer.py` (217 L, 1 dataclass + 1 pure function).
**Paper sources :** §3 (SARL), Table 11 hyperparams (GAMMA=0.999, lr_first=0.00025, lr_second=0.0002,
target_update=1000, α_EMA=0.45), eq.12 (tied-weight reconstruction with bias).
**Student sources :** `external/paper_reference/sarl_maps.py:663-887` (`train()` function, inclut la
branche CL scope sarl_cl dans notre port).

**Callers du port :**
- `sarl/training_loop.py:300` (appel par update step, setting 1-6).
- Tests parity `tests/parity/sarl/test_tier3_update.py`.

**DoD D.6** : doc créé, 11 findings tabulés, cross-ref avec 6+ D-sarl-* existants, 0 code touché.

---

## Architectural split (D.6-a)

Port **découpe intentionnellement** le student `train()` en 2 :
- **`sarl/trainer.sarl_update_step`** : branche non-CL (settings 1-6 SARL simple).
- **`sarl_cl/trainer`** : branche CL avec teacher + distillation + feature loss (deferred D.17+).

Docstring module L6-14 justifie : garder la CL branch ici "would tangle parity with
EWC/feature-preservation logic that Sprint 04b doesn't touch".

✅ **Split paper-preserving**, math SARL identique.

## Constantes module-level (D.6-b)

```python
GAMMA = 0.99           # L54 — DQN discount factor
CAE_LAMBDA = 1e-4      # L57 — Jacobian regularizer weight
```

### 🆘🆘 D-sarl-gamma (déjà tracé, D.7 fix queued)

**Paper Table 11** : `γ = 0.999`.
**Student `sarl_maps.py:L104`** : `GAMMA = 0.99` (**student diverge paper**).
**Port** : `GAMMA = 0.99` hardcoded (matches student, **not paper**).

**Impact** : `γ=0.99` vs `γ=0.999` change l'horizon temporel de la Bellman-equation. Sur 5M frames
MinAtar, l'effective horizon est `1/(1-γ)` = 100 steps (student) vs 1000 steps (paper). L'agent
paper voit beaucoup plus loin dans le futur → potentiellement change les Q-values apprises et
donc la z-score de reproduction.

**Fix queued D.7** : rendre `GAMMA` config-driven + default paper value 0.999.

### CAE_LAMBDA = 1e-4

Student `sarl_maps.py:L757` : `lam = 1e-4` inline. Paper Table 11 ne spécifie pas explicitement
cette valeur (paper silent ; `1e-4` est la valeur default Rifai 2011). ✅ **Port = student,
paper-consistent implicite**.

## `SarlUpdateOutput` dataclass (D.6-c)

Port L60-71 : 4 champs `(loss, loss_second, q_values, wager_logits)`. meta=False → les 3 derniers
sont None. Uniforme → caller ne branche pas sur arity.

✅ **Port addition safe** (student retourne tuple de taille variable selon `meta` + `previous_loss`).

## `sarl_update_step` math (D.6-d)

Le cœur du training step. Port L74-217 vs student L683-799 (non-CL path) :

| Step | Student                                                                  | Port                                                                    | Match |
|:----:|:-------------------------------------------------------------------------|:------------------------------------------------------------------------|:-----:|
| 1    | `cascade_rate_1 = 1.0/cascade_iterations_1`                              | idem L137                                                                | ✅    |
| 2    | `optimizer.zero_grad()` + `optimizer2.zero_grad() if meta` (L703-705)   | idem L145-147                                                           | ✅    |
| 3    | `batch_samples = transition(*zip(*sample))` (L708)                       | `Transition(*zip(*sample, strict=True))` (L150)                         | ✅ (port ajoute `strict=True` safer) |
| 4    | `states = torch.cat(batch_samples.state)` + siblings                     | idem L151-155                                                           | ✅    |
| 5    | `targets_wagering = target_wager(rewards, alpha)` (L717)                 | `target_wager_fn(rewards, alpha)` (L157) — injecté pour tests           | ✅    |
| 6    | 1st-order cascade forward loop L722-723 (sur `policy_net`)               | idem L160-163                                                           | ✅    |
| 7    | `Q_s_a = output.gather(1, actions)` (L735)                               | `q_s_a = q_policy.gather(1, actions)` (L164)                            | ✅    |
| 8    | non-terminal filter (L739-740)                                            | idem L167-172                                                            | ✅    |
| 9    | target net cascade forward (L744-750)                                    | idem L175-180                                                           | ✅    |
| 10   | `target = rewards + GAMMA * Q_s_prime_a_prime` (L753)                    | `td_target = rewards + GAMMA * q_s_prime` (L182)                        | ✅    |
| 11   | `W = policy_net.state_dict()['fc_hidden.weight']` (L756)                 | idem L185 (C.7 confirme state_dict() détache par défaut)                | ✅    |
| 12   | `loss = CAE_loss(W, target, Q_s_a, h1, lam)` (L799, non-CL path)        | `loss = cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)` (L188)           | ✅    |
| 13   | meta branch cascade forward L806-807                                     | idem L192-193                                                           | ✅    |
| 14   | `loss_second = binary_cross_entropy_with_logits(wager, targets_wagering)` | idem L194                                                                | ✅    |

✅ **14 étapes bit-parity avec student non-CL path.**

## (e) 🚨 Backward pass order — **load-bearing, documented**

Docstring module L18-31 décrit précisément le non-trivial ordre :

```
loss_second.backward(retain_graph=True)   # ① gradients via comparison_1 populate BOTH
                                           #    second_order_net.grad AND policy_net.grad
optimizer2.step()                          # ② update second-order
loss.backward()                            # ③ ADD main-task gradients into policy_net.grad
optimizer.step()                           # ④ update policy_net with ①+③ sum
scheduler1.step(); scheduler2.step()       # ⑤ LR decay
```

### Pourquoi "load-bearing" ?

- **Étape ①** : `loss_second` dépend de `wager` → `wager` dépend de `comparison_out` →
  `comparison_out` dépend de `comparison_1` (input au 2nd-order) → `comparison_1` dépend de
  `policy_net` (via `SarlQNetwork` tied-weight decoder). Donc `loss_second.backward()` traverse
  le graph jusqu'à `policy_net.grad`.
- **Étape ②** : `optimizer2` ne touche que `second_order_net.parameters()`. Le `policy_net.grad`
  populé par ① reste intact.
- **Étape ③** : `loss.backward()` **ajoute** aux grads existants (accumulation standard PyTorch).
  `policy_net.grad` contient maintenant ① + ③.
- **Étape ④** : `optimizer.step()` update `policy_net` avec la somme des gradients 1st-order et
  2nd-order.

**Conséquence** : **Le 2nd-order loss influence aussi le 1st-order weights**. Couplage essentiel
pour la métacognition paper. Swap ①/③ ou use un seul `.backward()` sur `loss + loss_second` →
optimizer2 ne verrait pas de gradient 2nd-order propre pour update, et la décomposition serait
perdue.

✅ **Student (L ~860-885) même ordre.** Port bit-parity. **Tier 3 parity test guard**
(`tests/parity/sarl/test_tier3_update.py`) vérifie cette invariance numériquement.

⚠️ **D-sarl-backward-order** (existing deviation, paper silent on this) : gardé en info, pas de
fix — c'est le comportement student, et on n'a pas de source paper qui prescrit un autre ordre.

## (f) D-sarl-recon-bias — **paper eq.12 vs student/port divergence**

**Paper eq.12** (paper_equations_extracted.md, SARL §3) :
```
Ŷ^(1) = ReLU(fc_hidden · Hidden + b_recon)
```
Le tied-weight decoder devrait avoir un **bias term** `b_recon` additionnel (learnable).

**Student `sarl_maps.py:L176`** :
```python
Output_comparison = f.relu(f.linear(Hidden, self.fc_hidden.weight.t()))  # NO bias
```

**Port `sarl/model.py:91`** :
```python
reconstruction = F.relu(F.linear(hidden, self.fc_hidden.weight.t()))  # NO bias
```

→ **Paper prescrit b_recon learnable, student omet, port matches student** (paper-vs-student
divergence 🆘).

**Impact** : `ReLU(W^T·h)` = 0 pour les coords où `W^T·h ≤ 0`. Sans bias, le decoder ne peut pas
shifter cette relu. Avec bias, chaque output peut saturer à 0 à un seuil appris.

**Fix scope D.7** (ou dedicated fix sub-phase) :
- Option A (paper-faithful) : ajouter `self.b_recon = nn.Parameter(torch.zeros(NUM_LINEAR_UNITS))`
  dans `SarlQNetwork.__init__` → `F.relu(F.linear(hidden, ..., bias=self.b_recon))`.
  **Impact init RNG** : 1 nouveau parameter → consomme RNG dans la séquence init → **brise
  parity avec student** (decorrélé du paper Table 11 existant).
- Option B (student-faithful) : rester sans bias, documenter divergence paper.

Policy 2026-04-19 "paper = source of truth" → **Option A**. Mais parity Tier 1 forward test devra
être **mis à jour** (les poids init diffèrent après ajout `b_recon`).

**Piste D.7 fix queued** : confirmer Option A avec Rémy avant D.7 impl.

## (g) Performance notes (hors parity)

### `non_terminal_idx` construction L167-172

Port (idem student) :
```python
non_terminal_idx = torch.tensor(
    [i for i, done in enumerate(is_terminal) if done == 0],
    dtype=torch.int64,
    device=device,
)
```

**Python list comprehension + host-device transfer** = slow-ish. Pour batch_size=32, négligeable.
Alternative pure-torch :
```python
non_terminal_idx = (is_terminal.squeeze(-1) == 0).nonzero(as_tuple=True)[0]
```

→ Bit-parity ? **OUI** (même indices retournés, même int64 dtype). Gain compute marginal sur
B=32, visible sur B>>32. **Piste D.7 fix queued** (plan L532 mentionne "non_terminal_idx
optimization").

### Cascade loop cost (D.4 re-emphasized)

Ici, le cascade loop **1st-order** (L160, L176) s'applique **à chaque update step** (training_freq
default = 4 frames/update → 1.25M updates sur 5M frames). Chaque update = 2 cascade loops
(policy + target) × 50 iters = 100 forwards. Soit **~125M forwards gaspillés** sur la phase
training (comparé à 250M sur rollout). D.4 log warning couvre déjà côté training_loop.

## (h) `target_wager_fn` injection — port addition safe

Port L87 : `target_wager_fn: Any` injecté comme param. Student appelle `target_wager` directement
(module global). Gain : tests peuvent passer un mock → parity test tier 3 utilise la
référence verbatim (`tests/parity/sarl/_reference_sarl.py:189`).

✅ **Non-breaking**.

## (i) Asserts meta-mode (L132-135)

Port ajoute 3 asserts d'invariant :
```python
if meta:
    assert second_order_net is not None
    assert optimizer2 is not None
    assert scheduler2 is not None
```

Student : pas d'assert ; si `meta=True` et pas de `second_order_net`, erreur tardive cryptique.
Port **fail-fast** explicitement. ✅ **Gain défensif**.

## Cross-reference Phase B audit deviations

| ID existant                  | Status D.6 review                                                |
|:-----------------------------|:-----------------------------------------------------------------|
| D-sarl-gamma (🆘+❌)         | Re-confirmed L54. Fix queued D.7 (config-driven + value 0.999)  |
| D-sarl-lr-2nd (🆘+❌)        | N/A in trainer.py (lr is caller responsibility via optimizer). D.9 config scope |
| D-sarl-adam-beta1/2 (🆘+❌) | N/A in trainer.py. D.9 scope                                     |
| D-sarl-sched-step (🆘+❌)   | scheduler step() called L203/216 — fréquence OK, step_size scope D.9 |
| D-sarl-target-update (🆘)    | N/A in trainer.py (target net update is caller sync in training_loop) |
| D-sarl-recon-bias (🆘+❌)   | Confirmed missing L91 sarl/model.py. **D.7 fix needed** (Option A) |
| D-sarl-backward-order (⚠️) | Documented L18-31 module docstring + tier 3 parity test         |
| D-sarl-bce-shape (⚠️)       | Confirmed L194 wager shape (B,2) + targets shape (B,2)          |
| D-sarl-dropout-position (🆘+⚠️) | 2nd-order cascade scope — sarl/model.py:144 — D.4 handled paper-Option A |

### Aucune nouvelle deviation surfacée en D.6 — toutes les découvertes sont déjà tracées.

## Fixes identifiées D.6

| ID       | Fix                                                                | Scope                              | Effort |
|:---------|--------------------------------------------------------------------|------------------------------------|:------:|
| D6→D.7   | `GAMMA` config-driven + paper value 0.999                          | plan D.7                           | queued |
| D6→D.7   | `non_terminal_idx` vectorisation pure-torch (perf, bit-parity)     | plan D.7                           | queued |
| D6→D.7   | **Add `b_recon` bias to tied-weight reconstruction** (Option A, paper-faithful) | sarl/model.py:91 + parity tests update | queued |
| D6-fix-1 | (skip) docstring module — déjà excellent, rien à ajouter           | —                                  | skip   |

## Résumé — `sarl/trainer.py`

- ✅ **Split architectural** : non-CL path ici, CL path dans sarl_cl (D.17 scope).
- ✅ **14 étapes `sarl_update_step` bit-parity** avec student `train()`.
- ✅ **Backward order load-bearing** : docstring explicite, tier 3 parity test guard.
- ✅ **`W = state_dict()['fc_hidden.weight']`** : détaché par défaut (C.7 validé).
- 🆘 **D-sarl-gamma** : re-confirmé, fix D.7.
- 🆘 **D-sarl-recon-bias** : re-confirmé, **décision Option A (paper-faithful) à acter en D.7**,
  impact parity tests Tier 1.
- ⚠️ **Perf `non_terminal_idx`** : vectorisable, bit-parity preservée, D.7 queued.
- ✅ **`target_wager_fn` injection + meta asserts** : gains port non-breaking.
- **0 nouvelle deviation, 3 fixes queued D.7 (GAMMA, non_terminal_idx, b_recon).**

**D.6 clôturée. 0 code touché. 3 fixes pour D.7 batch (le plus dense à venir).**
