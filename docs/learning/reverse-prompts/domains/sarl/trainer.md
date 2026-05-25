# Reverse-prompt — `src/maps/domains/sarl/trainer.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl/trainer.py`
**Taille :** ~220 lignes — pure function `sarl_update_step`.
**Paper :** §3 + Table 11.
**Review existant :** `docs/reviews/sarl-trainer.md`.

---

## 1. Quelle aurait été la spec ?

> Écris une **pure function** `sarl_update_step(...)` qui implémente
> un seul DQN update step + le metacognitive branch optionnel. Port
> de `maps.py:663-887` (standard-DQN path only, pas la branche CL).
>
> **Pas de globals** — tout passé en args :
> - `sample: list[Transition]` (depuis buffer)
> - `policy_net, target_net` (Q-networks, target frozen par caller)
> - `second_order_net` (None si meta=False)
> - `optimizer, optimizer2` (Adam pour les 2)
> - `scheduler1, scheduler2` (StepLR)
> - `meta: bool` — gate du metacognitive branch
> - `alpha, cascade_iterations_1, cascade_iterations_2, gamma, device`
> - `target_wager_fn` (injection callable pour tests)
>
> **Returns `SarlUpdateOutput`** dataclass uniforme :
> ```python
> @dataclass
> class SarlUpdateOutput:
>     loss: Tensor                       # 1st-order, always
>     loss_second: Tensor | None = None  # 2nd-order, only if meta
>     q_values: Tensor | None = None
>     wager_logits: Tensor | None = None
> ```
>
> **Inner steps (paper order, locked for parity)** :
>
> 1. `optimizer.zero_grad()`, `optimizer2.zero_grad()` if meta
> 2. Unpack batch transitions via `Transition(*zip(*sample, strict=True))`
> 3. `targets_wagering = target_wager_fn(rewards, alpha)`
> 4. **1st-order forward (cascade)** : `cascade_iterations_1` iters de
>    `q_policy, h1, comparison_1, main_task_out = policy_net(states,
>    main_task_out, cascade_rate_1)`
> 5. `q_s_a = q_policy.gather(1, actions)` (Q for action taken)
> 6. **TD target** : compute non-terminal mask, run target_net forward
>    on non-terminals (with cascade), `q_s_prime = max(target).detach()`
> 7. `td_target = rewards + gamma * q_s_prime`
> 8. `W = policy_net.state_dict()["fc_hidden.weight"]` (live weight view,
>    state_dict default detaches)
> 9. **1st-order loss** : `loss = cae_loss(W, td_target, q_s_a, h1,
>    CAE_LAMBDA=1e-4)` — Huber recon + Jacobian
> 10. **If meta** :
>     - 2nd-order forward (cascade) : `wager, comparison_out =
>       second_order_net(comparison_1, comparison_out, cascade_rate_2)`
>     - `loss_second = F.binary_cross_entropy_with_logits(wager,
>       targets_wagering)`
>     - **Critical ordering** :
>       ```
>       loss_second.backward(retain_graph=True)   # ① grad → policy via comparison_1
>       optimizer2.step()                          # ② 2nd-order weights move
>       loss.backward()                            # ③ ADDS to policy grad
>       optimizer.step()                           # ④ policy_net consumes ①+③
>       ```
>     - `scheduler1.step(); scheduler2.step()`
> 11. **Else (meta=False)** : `loss.backward(); optimizer.step();
>     scheduler1.step()`

## 2. Contraintes scientifiques

### Two-loss gradient pattern (paper-faithful order)

Quand `meta=True`, le gradient flow est subtil :

- `loss_second` dépend de `wager` qui dépend de `comparison_1` qui
  dépend de `policy_net.fc_hidden` (via le tied/dedicated decoder)
- Donc `loss_second.backward(retain_graph=True)` **populate** des
  gradients dans `policy_net.parameters()` ET `second_order_net.parameters()`
- `optimizer2.step()` ne touche QUE les params 2nd-order
- `loss.backward()` AJOUTE plus de gradient à `policy_net`
- `optimizer.step()` consume LA SOMME des gradients de loss_2 + loss_1

**Sémantique** : `policy_net` apprend simultanément à :
- Prédire Q correctement (loss_1 = Huber TD)
- Faire en sorte que sa reconstruction (comparator) soit
  informative pour le wager (loss_2 = BCE wager target)

**C'est ce qui distingue MAPS de DQN+attention naïf** : le 1st-order
est explicitly co-trained avec le 2nd-order.

### `CAE_LAMBDA = 1e-4` hardcoded

Pas exposé en config. Could be (cf. `losses.md` Q2). Hardcoded
par parity student.

### `non_terminal_idx` vectorisé

Student fait `[i for i, t in enumerate(is_terminal) if t == 0]`
(list comp). Vectorisé en `(is_terminal.view(-1) == 0).nonzero(as_tuple=True)[0]`
(int64 tensor). Bit-parity (Sprint-08 D.7 confirmed via test_tier3_update).

### `target_wager_fn` injection

Le `target_wager` est passé comme callable, pas importé directement.
Permet aux tests de swap la reference implementation pour parity.

## 3. Contraintes d'ingénierie

### Pure function = testable

Pas de side effects sauf via les `.step()` calls (qui mutent les
optimizer states explicitement passés). Tests Tier-3 peuvent assert
bit-exact contre `_reference_sarl.py` après load_state_dict + même
sample batch.

### `cascade_rate = 1.0 / cascade_iterations`

Convention : si `cascade_iterations_1=50`, `cascade_rate_1=0.02`.
Si `=1`, `cascade_rate_1=1.0` (no-op cascade, single forward). Cohérent
avec `core/cascade.n_iterations_from_alpha(α)`.

### `W = state_dict()["fc_hidden.weight"]` détache

`state_dict(keep_vars=False)` (default) retourne tensors détachés.
Donc le Jacobian term dans `cae_loss` ne propage PAS gradient
directement vers `fc_hidden.weight` — seulement via `h`'s backward
path. Subtilité documented in `core/losses.md`.

### `retain_graph=True`

Nécessaire car le graph computation est utilisé pour 2 backwards
(loss_second puis loss). Sans retain, le 2e backward crash car le
graph a été free.

### `optimizer2.zero_grad()` AVANT loss_second

Logique standard : clear grad before backward. Mais `optimizer.zero_grad()`
au tout début aussi (avant le forward). Donc quand loss_second.backward
populate policy_net.grad, il commence à 0 (clean). Et quand
loss.backward s'ajoute, il s'ajoute proprement.

## 4. Deviations / dettes présentes

### 🚨 D-sarl-backward-order (load-bearing, paper-faithful)

L'ordre `loss_second.backward → optimizer2.step → loss.backward →
optimizer.step` est **load-bearing**. Swap et tu changes la
trajectoire d'apprentissage.

Documenté dans le module docstring. Tier-3 parity test guard contre
les régressions.

### D-sarl-bce-shape (doc-only)

`F.binary_cross_entropy_with_logits(wager[B,2], targets[B,2])` —
shape ok. Paper eq.5 dit scalaire `y` mais code utilise (B, 2) one-hot.
Mathématiquement équivalent (2 sigmoids indépendants). Voir
`core/second_order.md`.

### D-sarl-recon-bias (D.7 resolved)

`b_recon` added zero-init. Tier 1/3 parity preserved at init.

### Pas de DETTE active sur ce module

## 5. Questions ouvertes

- **Q1 :** Le pattern `loss_second.backward → optim2.step → loss.backward
  → optim.step` est-il **différent** de la version où on fait
  `(α·loss_1 + β·loss_2).backward(); optim.step(); optim2.step()` ?

  Théoriquement :
  - Version actuelle : optim2 voit grad_2 only ; optim voit grad_2 +
    grad_1
  - Version single-backward : optim voit α·grad_1 + β·grad_2 ; optim2
    voit α·grad_1 + β·grad_2 (mais que sur params 2nd-order)

  La différence : dans v1, optimizer2 ne voit JAMAIS le grad de loss_1,
  alors qu'en single-backward, il en voit la contribution (même si
  loss_1 ne touche pas les params 2nd-order, le backward peut faire
  des side-effects via Adam moment terms).

  **À vérifier mathématiquement** : ces 2 patterns sont-ils
  équivalents ?

- **Q2 :** `CAE_LAMBDA = 1e-4` hardcoded. Pourquoi pas en config ?
  Probablement parce que aucune ablation l'a varié. Mais c'est un
  magic number qui pourrait surprendre un futur lecteur.

- **Q3 :** Le `target_wager_fn` injection est belle pour les tests
  mais ajoute un param. Alternative : import direct + monkey-patch
  dans les tests. Trade-off lisibilité vs flexibilité.

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Pure function design (no globals)
- Statement order exact (paper-faithful)
- `SarlUpdateOutput` dataclass uniforme
- `target_wager_fn` injection
- `retain_graph=True` (nécessaire)
- `CAE_LAMBDA = 1e-4` (pour l'instant)

### À améliorer
- **Documenter clairement le gradient flow** : ajouter un schéma
  ASCII dans la docstring montrant qui flow où
- **Type hints `Any` pour target_wager_fn** → typed
  `Callable[[Tensor, float], Tensor]`
- **Exposer `CAE_LAMBDA`** en config si une ablation le demande
- **Vérifier `optimizer2.zero_grad()` est appelé une seule fois**
  (avant le forward, pas avant chaque cascade iter)

### Tests à écrire
- Parity bit-exact vs `_reference_sarl_update_step` (Tier 3)
- `meta=False` skip la 2nd-order branch
- `meta=True` calcule loss_second
- Output dataclass shape : loss scalar, loss_second scalar, q_values
  (B, A), wager_logits (B, 2)
- Gradient flow : after `loss_second.backward(retain_graph=True)`,
  `policy_net.fc_hidden.weight.grad` non-zero
- `target_wager_fn` injection : mock + assert called once

### Opportunités perf identifiées
- **Cascade no-op skip sur 1st-order Q-net** : si meta off ou path
  déterministe, replace 50-iter loop par 1 forward. 50× local speedup
  pour setting 2/4 fwd phase. Tier-3 parity exigeant.
- **Async target_net sync** : actuellement load_state_dict bloquant.
  Marginal mais easy win.
- **Batched non_terminal indexing** : déjà vectorisé. ✅

### Connexion
- Utilise : `data.Transition`, `losses.cae_loss`
- Appelé par : `training_loop.run_training` (in inner loop)
- Tests Tier-3 dans `tests/parity/sarl/test_tier3_update.py`
