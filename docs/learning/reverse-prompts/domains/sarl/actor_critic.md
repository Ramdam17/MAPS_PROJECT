# Reverse-prompt — `src/maps/domains/sarl/actor_critic.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl/actor_critic.py`
**Taille :** ~510 lignes.
**Status :** **ACB = Actor-Critic Baseline, paper Setting 7**.
**Paper :** §3.2 + Tables 6/7. Young & Tian (2019) arXiv:1903.03176.

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `actor_critic.py` qui implémente AC(λ) — un
> **algorithme structurellement différent** de DQN+meta+cascade, qui
> sert de baseline non-MAPS pour les Tables 6/7. Porte
> `external/paper_reference/sarl_ac_lambda.py` (Young & Tian 2019,
> Algorithm 1).
>
> **Pas de cascade**, **pas de 2nd-order**, **pas de replay buffer**.
> Online actor-critic avec eligibility traces. Strictement séparé du
> reste SARL.
>
> **Classes** :
>
> **1. `ACNetwork`** : Conv(16) → FC(128) → (policy softmax, value
> linear). **SiLU / dSiLU activations** (Elfwing et al. 2018, recommandés
> pour value-based RL).
>
> **2. `ACBConfig` dataclass** :
> - game, seed, num_frames, output_dir
> - α=0.000488 (1/2048), λ=0.8 (trace decay), γ=0.99, β=0.01 (entropy)
> - γ_rms=0.999, ε_rms=0.0001, min_denom=1e-4
> - validation_every_episodes=500 (15 pour freeway), validation_episodes=2
> - device, log_every_episodes
> - `for_game(game, **kwargs)` classmethod : applique freeway override
>   (`validation_every_episodes=15`) unconditional
>
> **3. `ACBTrainer`** : orchestrator online AC(λ).
> - `__init__(env, network, cfg)`
> - `train()` : boucle online, no buffer
> - `_update(...)` : custom RMSprop avec debiasing
> - `validate()` : greedy rollouts
>
> **Distinctives algorithmiques** :
> - **Custom RMSprop avec init debiasing** :
>   ```
>   MSG_t = γ_rms · MSG_{t-1} + (1 - γ_rms) · grad²
>   param += α · grad / √(MSG_t / (1 - γ_rms^(t+1)) + ε_rms)
>   ```
>   Le `(1 - γ_rms^(t+1))` debiasing **N'EST PAS** dans
>   `torch.optim.RMSprop`. Doit être implémenté manuellement, sinon
>   diverge de reference.
> - **Eligibility traces** TD(λ) replacing-trace style
> - **Entropy bonus** `β·H(π)` ajouté à la loss policy
> - **Numerical floor** `MIN_DENOM=1e-4` dans `log(π + MIN_DENOM)`
>   pour éviter `log(0)`. Pas dans paper Table 11 — D-sarl-acb-min-denom.

## 2. Contraintes scientifiques

### Pourquoi un baseline AC(λ) ?

Pour faire un "fair comparison", il faut comparer MAPS à **un
algorithme RL standard**. AC(λ) Young & Tian 2019 :
- C'est l'algo recommandé par les auteurs de MinAtar pour leur testbed
- C'est très bien rodé sur ces tâches (papiers de référence)
- Il est **structurellement différent** de DQN (pas de Q-fonction, pas
  de target net, pas de replay buffer) → contraste maximal avec MAPS

Si MAPS-DQN bat AC(λ), c'est un signal fort (vs comparer MAPS-DQN à
DQN-vanilla qui aurait juste démontré que l'ajout MAPS est meilleur
que rien).

### Sutton & Barto §13.5 (eligibility traces)

```
δ_t = R_t + γ·V(S_{t+1}) - V(S_t)   # TD error
e_t = γ·λ·e_{t-1} + ∇V(S_t)          # eligibility trace
θ_{t+1} = θ_t + α·δ_t·e_t            # update
```

Replacing-trace : remplacer `e_t = γ·λ·e_{t-1} + ∇V` par `e_t = max(γ·λ·e_{t-1},
∇V)` (ou similaire). Évite l'accumulation infinie sur des features
fréquentes.

### SiLU / dSiLU

`SiLU(x) = x · σ(x)` (Sigmoid Linear Unit), équivalent
`F.silu`. **dSiLU** = `σ(x) · (1 + x·(1-σ(x)))` (dérivée de SiLU).

Le code utilise **2 sigmoid calls** dans `dsilu` (pas cached). Si on
optimise en cachant `s = sigmoid(x)`, on change l'ordre des operations
flottantes → drift ~1 ULP float32 par backward. **Refactor breaks
parity**, ne pas toucher.

### `min_denom = 1e-4` dans `log(π + ε)`

Pas dans paper Table 11. Reference le met explicitement. Évite
`log(0) = -inf → NaN` quand `π` est très petit. Tracked as
D-sarl-acb-min-denom (info only, pas un fix).

## 3. Contraintes d'ingénierie

### Custom RMSprop manual update

Le `(1 - γ_rms^(t+1))` debiasing n'est pas dans `torch.optim.RMSprop`.
Donc on ne peut pas utiliser PyTorch optimizer ; on doit faire
manuellement :

```python
for param, grad in zip(model.parameters(), grads):
    msg = gamma_rms * msg + (1 - gamma_rms) * grad**2
    debias = msg / (1 - gamma_rms**(t+1))
    param += alpha * grad / sqrt(debias + eps_rms)
```

Note : `t` est le compteur global, pas le param-specific Adam step.
Donc le debiasing est synchronized across params.

### `Transition` namedtuple différent

ACB n'utilise pas le `Transition` de `data.py` (pas de replay buffer
shared). Son propre namedtuple : `(state, last_state, action, reward,
is_terminal)`. Différent ordre / différents fields. Pas de
confusion possible (modules séparés).

### `for_game(game, ...)` factory

```python
@classmethod
def for_game(cls, game, **kwargs):
    if game == "freeway":
        kwargs["validation_every_episodes"] = 15  # unconditional override
    return cls(game=game, **kwargs)
```

Freeway a des épisodes très longs (game ≈ minutes IRL). Si on validate
tous les 500 episodes comme les autres games, on attend trop. Override
à 15. **Unconditional** (pas `setdefault`) car `run_sarl.py` toujours
passe `validation_every_episodes` (sinon setdefault no-op).

### Entropy bonus

`policy_loss = -log(π(a)) · δ + β·H(π)`. β=0.01. Encourage
exploration. Standard actor-critic technique.

## 4. Deviations / dettes présentes

### ⚠️ D-sarl-acb-min-denom (info only)

`MIN_DENOM=1e-4` ajouté dans `log(π)` pour stability. Pas dans paper
Table 11 mais dans reference code. Tracked pour transparency. Pas un
fix nécessaire.

### ✅ D-sarl-setting-7 (Sprint-08 D.31 resolved)

ACB ported as `actor_critic.py`, routed via `--setting 7` in CLI.
Bit-identical parity (4/4 tests in `test_acb_parity.py`). 5 games × 3
seeds completed.

### Pas de DETTE active sur ce module

Module séparé du reste SARL. Bien isolé. Pas de risk de cross-coupling.

## 5. Questions ouvertes

- **Q1 :** Si MAPS-DQN bat AC(λ) sur 4 games sur 5 (per Table 6),
  qu'est-ce que ça nous dit vraiment ? AC(λ) sample-efficient ≠ DQN
  sample-efficient. Comparison est non-trivial.

- **Q2 :** Le `dSiLU` double-sigmoid est un détail d'implémentation
  qui affecte parity. Y a-t-il d'autres pièges similaires dans le
  code (ordre des ops flottantes load-bearing) ?

- **Q3 :** Custom RMSprop debiasing — pourquoi pas utiliser
  `torch.optim.Adam` (qui a `bias_correction1, bias_correction2`) ?
  Adam debias les moments d'ordre 1 et 2 ; RMSprop debias juste
  l'ordre 2. Différent comportement.

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Custom RMSprop manual update with debiasing
- SiLU / dSiLU activations (with double sigmoid in dsilu — parity)
- Eligibility traces TD(λ) replacing-trace
- Entropy bonus β=0.01
- `min_denom=1e-4` in log(π) — paper-faithful via reference code
- `for_game(game)` factory with freeway override
- Separate `Transition` namedtuple

### À améliorer
- **Type hints `Optional[torch.Tensor]`** pour states avant init
- **Logging structuré** : per-episode JSON record
- **Test cuda compat** : ACBTrainer probablement CPU-only en pratique,
  mais double-check no hardcoded `device='cpu'`

### Tests à écrire
- Parity bit-exact vs `_reference_sarl_ac_lambda` (déjà existant)
- `for_game("freeway", **kwargs)` override
- Custom RMSprop debiasing : valeurs correctes après N steps
- Eligibility trace decay : `e_t = γ·λ·e_{t-1} + ∇V`
- Entropy bonus dans policy loss

### Opportunités perf
- **GPU pour ACNetwork forward** : probable, mais env steps CPU
  limitent
- **Vectoriser eligibility traces** : actuellement per-param scalar
  ops dans `_update`. Stack the params, vectorize.
- **Caching `sigmoid(x)` dans `dsilu`** : 2× sigmoid call → 1× call.
  **Casse parity 1 ULP** — pas faire sauf si Tier-1 parity test
  ajusté.

### Connexion
- Utilise : `torch`, `nn` only (pas de cascade, pas de losses MAPS)
- Appelé par : `domains/sarl/cli.py` quand `--setting 7`
- Tests parity dans `tests/parity/sarl/test_acb_parity.py`
