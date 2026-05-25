# Reverse-prompt — `src/maps/domains/marl/act.py`

**Module path actuel :** `src/maps/experiments/marl/act.py`
**Taille :** ~116 lignes.
**Discrete action space only (E.5 scope lock).**

---

## 1. Quelle aurait été la spec ?

> Écris l'action head pour MARL. MeltingPot substrates exposent
> **Discrete** action spaces seulement. Port student
> `external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/act.py`
> mais drop tout sauf le Discrete branch (Box, MultiBinary,
> MultiDiscrete, Mixed → not needed).
>
> **3 classes** :
>
> **1. `FixedCategorical`** (extends `torch.distributions.Categorical`)
> avec student shape conventions :
> - `sample()` return extra trailing dim (unsqueeze(-1))
> - `log_probs(actions)` sum log-probs across last dim
> - `mode()` argmax + trailing dim
>
> **2. `Categorical`** (nn.Module) :
> - `Linear(num_inputs, num_outputs)` avec init orthogonal/xavier,
>   gain=0.01 (action heads usually small init)
> - `forward(x, available_actions=None)` → `FixedCategorical(logits=...)`
> - Si `available_actions` provided, mask invalid actions à `-1e10`
>
> **3. `ACTLayer`** wrapper :
> - `__init__(action_space, inputs_dim, use_orthogonal, gain)`
> - Raise NotImplementedError si action_space n'est pas Discrete
> - `forward(x, available_actions=None, deterministic=False)` →
>   `(actions, action_log_probs)`
> - `evaluate_actions(x, action, available_actions=None,
>   active_masks=None)` → `(action_log_probs, dist_entropy)` pour PPO
>   update

## 2. Contraintes scientifiques

### Categorical action distribution

Discrete action = sample from Categorical(logits=π(s)). Plus l'entropy
bonus β·H(π) pour exploration.

### Initialisation `gain=0.01`

Standard pour action heads : init petit pour que la softmax soit
proche d'uniforme au début (exploration). Sinon le réseau a un
strong prior over actions dès l'init.

### `active_masks` for dead agents

Pendant le PPO update, certains agents peuvent être morts (terminal).
Le `active_masks` permet de ne compter que les samples actifs dans
l'entropy compute :
```python
dist_entropy = (entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
```

## 3. Contraintes d'ingénierie

### Strict Discrete-only

```python
if action_space.__class__.__name__ != "Discrete":
    raise NotImplementedError(...)
```

Si futur MeltingPot expose des Box (continuous) actions, faudrait
porter le branch Box.

### Student shape convention

`FixedCategorical.sample()` returns `(B, 1)` au lieu de `(B,)`. C'est
pour matcher la convention "action a un trailing dim" du buffer.

Idem pour `log_probs(actions)` qui squeeze + view + sum + unsqueeze.

### `available_actions` masking via `-1e10`

```python
logits[available_actions == 0] = -1e10
```

Force softmax à donner ~0 probabilité aux actions invalides. Standard
pattern.

## 4. Deviations / dettes présentes

Pas de déviation paper.

**Pas de DETTE active.**

## 5. Questions ouvertes

- **Q1 :** Le `-1e10` mask peut produire NaN dans softmax si tous les
  actions sont masked. Edge case ? Devrait raise.

- **Q2 :** `gain=0.01` est aggressive. Pas de comparison avec
  `gain="relu"` ou `gain=1.0` standard ?

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Discrete-only branch
- `FixedCategorical` shape conventions
- `-1e10` mask
- Orthogonal init + gain=0.01

### À améliorer
- **Assert no-all-masked** : raise si all available_actions=0
- **Continuous (Box) support** futur si needed

### Tests à écrire
- Sample shape (B, 1)
- log_probs shape (B, 1)
- mode() deterministic
- Available actions mask works
- Active masks weighted entropy

### Connexion
- Utilisé par : `policy.{MAPPOActor, MAPSActor}`
- Utilise : `util.init`
