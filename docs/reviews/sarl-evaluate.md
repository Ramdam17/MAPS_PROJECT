# Component review — `src/maps/experiments/sarl/evaluate.py`

**Review sub-phase :** Sprint-08 D.10.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl/evaluate.py` (253 L, 2 dataclasses + 1 Protocol
+ 2 functions).
**Paper sources :** §3 (SARL). Paper Tables 5/6 : mean returns (aggregated across seeds).
**Student sources :** `external/paper_reference/sarl_maps.py:890-1068` (`evaluation()`).

**Callers du port :**
- `sarl/training_loop.py:385` (validation cadence every N episodes).

**DoD D.10** : doc créé, 5 findings documentés, 0 code touché.

---

## Architectural split student → port (D.10-a)

Student `evaluation(env, meta, t, ..., r_buffer)` mêle 3 responsabilités :
1. Rollout sous ε-greedy policy actuel (continue d'explorer pendant eval).
2. Appel `train(..., train_or_test=False)` pour accumuler une val loss.
3. Partage le replay buffer training (muté pendant eval).

Port clean 2-function API :
- `run_greedy_episode` : **pure rollout greedy** (ε=0, no RNG consumption par exploration).
- `aggregate_validation` : N rollouts → `ValidationSummary`.

Docstring module L12-27 justifie : *"Mixing #1 and #2 means 'validation return' is coupled to
whatever ε is at frame t, which drifts. It also means evaluate() touches policy_net in eval mode
but uses ε-greedy action selection — an unprincipled mix."*

✅ **Split cleanup gain net** : val return non-drifté par ε.

### Impact paper parity
Les Tables 5/6 paper rapportent `mean(episodic_return)` across seeds. Student `evaluation()`
retourne un `G` cumulé équivalent au port `episodic_return`. **Math identique** ; seule la
policy change (student ε-greedy vs port greedy).

⚠️ **Divergence potentielle** : si paper Tables 5/6 ont été générées avec student `evaluation()`
ε-greedy → notre val greedy **peut différer légèrement** (ε ≈ 0.1 à t=100k → 10% random actions
→ val return potentiellement plus bas d'une fraction). Non-critique pour Tables mais à noter.

Docstring L36-38 note cette divergence : *"Mnih et al. (2015) uses fixed ε=0.05 for consistency;
we go further and set ε=0 (purely greedy) to remove any RNG drift from validation numbers."*

## Dataclasses (D.10-b)

### `EpisodeMetrics` (L67-77)
Fields : `episodic_return`, `n_steps`, `mean_wager_bet/nobet`, `bet_ratio`, `wager_logits_trace`.
- Wager stats populated uniquement si `second_order_net` provided.
- `wager_logits_trace` field default `list[np.ndarray]` — pour offline analysis (opt-in via
  `collect_wager_trace=True`).

### `ValidationSummary` (L80-89)
Fields : `mean_return, std_return, mean_steps, n_episodes, mean_bet_ratio`.

⚠️ **`std_return` convention** : `np.std(returns)` default `ddof=0` (population std divisor N).
Pour N=3 (paper SARL validation `n_episodes=3`), ddof=1 (sample std divisor N-1) aurait une
sémantique plus correcte. Différence numérique `std(ddof=0) / std(ddof=1) = sqrt((N-1)/N)` ≈
0.816 pour N=3. **Non-bloquant** ; student code ne spécifie pas non plus → parity.

**Piste D10-fix-1** (debate) : ajouter un commentaire pour expliciter la convention, ou passer
`ddof=1` si plus scientifiquement correct. Skip pour parity avec student par défaut.

## `run_greedy_episode` (D.10-c)

### Structure
1. Save `training` state + `policy_net.eval()` (same for second_order).
2. `env.reset()` + `get_state()`.
3. Loop `while not done and n_steps < max_steps` :
   - `greedy_action(state, policy_net, cascade_1)` → action tensor (D.5 hérité).
   - If meta : recompute policy forward + second_order forward for wager logits (softmax pour viz).
   - `env.act(action)` → reward, done.
   - Accumulate return, increment steps.
4. `finally:` restore training mode.

### (c1) `torch.softmax(wager_logits, dim=-1)` L179

Port applique softmax sur `wager_logits` pour l'eval viz, **mais** `SarlSecondOrderNetwork.forward`
retourne des raw logits (paper-faithful eq.3, C.6). Donc softmax ici est un **post-processing**
pour :
- `mean_wager_bet/nobet` : probs normalisées entre bet/no-bet pour viz.
- `bet_ratio` via `probs[0] >= probs[1]` (L182-183) — **monotonique** vis-à-vis des logits, donc
  équivalent à `logits[0] >= logits[1]`.

✅ **Cohérent avec C.6** (raw logits pendant training, softmax pour eval viz OK, pas de re-introduction
du bug softmax pré-C.6).

### (c2) 🚨 Duplicate forward dans le meta branch

Ligne 157 : `action = greedy_action(state, policy_net, cascade_1)` → forward policy 1× (cascade loop).
Lignes 168-171 : **re-forward** policy_net (cascade loop complet) juste pour récupérer
`comparison_1` (non-exposé par greedy_action qui ne retourne que l'action).

**Coût** : 2× le forward policy par step en meta eval. Sur 50 itérations cascade (settings 4/6),
c'est **100 forwards par step**.

Docstring L160-163 justifie : *"Small duplicate work — acceptable outside the training loop."*

**Vérification** :
- Eval freq : every 50 episodes × 3 episodes ~ 6% des runs SARL.
- Ce duplicate se produit par step **dans** l'eval path.
- Pour un run 5M frames avec eval cadence 50 ep : ~10k eval_steps × 50 cascade × 2 = 1M forwards
  duplicate → ~0.02% de l'overhead total. Effectivement négligeable.

✅ **Acceptable** mais documenter. Pas de fix.

### (c3) D.4 cascade no-op re-emphasized

Les cascade loops L168-171 (policy forward) appliquent le même no-op que training (D.4) — 50 iters
= 1 iter effectif car SarlQNetwork forward deterministic. Eval path hérite donc l'overhead de la
D.4 Option A.

La 2nd-order cascade loop L174-177 (si meta=True) **a du dropout** dans le forward (p=0.1), donc
**pas** de no-op sur ce path. Mais `second_order_net.eval()` L143 désactive le dropout → **en
eval le 2nd-order cascade AUSSI devient no-op** (pas de stochasticité à averager).

→ En eval mode : **les 2 cascade loops (1st + 2nd order) sont no-op**. Le greedy path pourrait en
principe shortcut à 1 iter sans changer le résultat. **Non-fait** (paper-faithful Option A).

Déjà couvert par D.4 log.warning au setup training ; hérité par eval. Rien à ajouter.

### (c4) `with torch.no_grad()` L167

Correctement wrapped pour le meta forward. Le `greedy_action` L157 utilise aussi un `torch.no_grad()`
interne (rollout.py D.5). ✅ Pas d'accumulation graph en eval.

### (c5) `max_steps = 100_000` safety cap

Garantit pas de boucle infinie si env déconne. Paper par défaut : MinAtar games terminate naturally.
**OK defensive coding.**

## `aggregate_validation(env, ..., n_episodes)` (D.10-d)

### Structure
- `n_episodes` appels à `run_greedy_episode(env, ...)`.
- Collect `returns`, `steps`, `bet_ratios`.
- Stats : `np.mean(returns)`, `np.std(returns)`, `np.mean(steps)`, `np.mean(bet_ratios)`.

### (d1) Env sharing entre épisodes

Le même `env` object est passé dans les N appels. `run_greedy_episode` fait `env.reset()` au début
→ state de l'env reset proprement. Par contre le **RNG state MinAtar** (via `np.random` global
legacy) continue sa séquence.

Sur N=3 eval episodes :
- Episode 1 : reset → consume RNG for initial state randomization.
- Episode 2 : reset → consume more RNG (continuation).
- Episode 3 : idem.

→ **Eval déterministe par rapport au seed global** (set_all_seeds au start), mais **couplé au
training RNG state**. Si on change le training ordre légèrement (ex: un update différent), l'eval
verra un state initial différent au next validation → eval returns varient.

**Piste D10-fix-2** (debate) : save/restore RNG state autour de l'eval, ou utiliser un eval-dedicated
RNG. Skip — student ne le fait pas non plus, parity.

### (d2) Aucun re-seed par épisode

Pas d'appel `set_all_seeds` entre les n_episodes. Student idem. ✅ Parity.

### (d3) Bet_ratios optional

`if m.bet_ratio is not None: bet_ratios.append(...)`. Filtré pour les n_steps=0 cases (pas de
step → pas de bet_ratio calculable). Rare mais défensif.

## Callers audit

| Caller                                | Usage                                                  | Status |
|:--------------------------------------|:-------------------------------------------------------|:------:|
| `sarl/training_loop.py:385`           | `aggregate_validation(env, policy_net, cascade_1, n_episodes=cfg.validation_iterations, ...)` | ✅ |

## Fixes identifiées D.10

| ID        | Fix                                                                    | Scope                          | Effort |
|:----------|------------------------------------------------------------------------|--------------------------------|:------:|
| D10-fix-1 | (debate) `std(ddof=1)` vs `ddof=0` — explicite ou aligner avec stats convention | `evaluate.py:247`  | skip (parity) |
| D10-fix-2 | (debate) Save/restore RNG around eval pour reproducibility independente du training | `aggregate_validation` | skip (parity) |

## Cross-reference reviews

- **D.4 cascade no-op** : hérité par eval greedy path (et 2nd-order en eval mode : dropout désactivé
  → 2nd-order cascade **aussi** no-op en eval). Couvert par D.4 logging au setup training. Rien
  à fix en D.10.
- **D.5 `greedy_action`** : utilisé L157, signature correcte.
- **C.6 raw logits wager** : respectée (softmax L179 est un post-processing viz, pas un bug
  pre-C.6).

### Aucune nouvelle deviation paper surfacée en D.10.

## Résumé — `sarl/evaluate.py`

- ✅ **Architectural split** cleanup student → port 2-function API. Gain net : val return non-drifté
  par ε.
- ✅ **`run_greedy_episode`** : structure propre, eval/train mode restore correctly, `torch.no_grad()`
  wrapped, `max_steps` safety cap.
- ✅ **Wager softmax L179** : post-processing viz-only, cohérent C.6 raw logits paper-faithful.
- ⚠️ **Duplicate forward** dans meta branch (documented, acceptable < 0.02% overhead total).
- ⚠️ **ε=0 eval** vs paper Mnih tradition ε=0.05. Divergence documentée, gain reproducibility.
- ⚠️ **`std(ddof=0)`** default convention, parity avec student.
- ⚠️ **RNG coupling** env state entre eval calls. Parity avec student.
- ✅ **D.4 cascade no-op inherited**, couvert par logging training setup.
- **0 nouvelle deviation paper, 0 bug, 2 debates minor skip.**

**D.10 clôturée. 0 fix direct, 2 skips debate. 0 code touché.**
