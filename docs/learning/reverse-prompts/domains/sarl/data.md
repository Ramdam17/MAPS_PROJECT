# Reverse-prompt — `src/maps/domains/sarl/data.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl/data.py`
**Taille :** ~280 lignes.
**Paper :** §3, Table 11 (α=0.45 EMA).
**Référence externe :** Mnih et al. (2015) replay buffer.

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `data.py` qui implémente le replay buffer
> SARL et les helpers d'état. Bit-exact parity avec
> `external/MinAtar/examples/maps.py` (commit `ec5bcb7`).
>
> **3 components** :
>
> **1. `Transition` namedtuple** : `(state, next_state, action, reward,
> is_terminal)`. Field order **fixed** (paper convention). NamedTuple
> for memory efficiency + immutability.
>
> **2. `SarlReplayBuffer`** : cyclic Python list-based replay buffer.
> - `__init__(buffer_size)` : `buffer_size=100_000` typique
> - `add(*args)` → `Transition(*args)`. Cyclic overwrite via
>   `(self.location + 1) % buffer_size` quand plein.
> - `sample(batch_size)` → `random.sample(buffer, batch_size)`.
>   **Important : utilise `random.sample` Python, pas
>   `np.random.choice` ni `torch.randperm`**. Sinon différent stream
>   RNG → différents transitions sampled → casse parity.
> - `__len__()` → len(buffer)
>
> **3. `get_state(s, device)`** : convert MinAtar numpy state
> `(10, 10, C)` → torch tensor `(1, C, 10, 10)` float. Permute +
> unsqueeze + float.
>
> **4. `target_wager(rewards, alpha)`** : EMA-based wager target
> generation pour le 2nd-order. **Crucial :**
> - `alpha` passé en pourcent (e.g. `alpha=45` pour 0.45)
> - Divise par 100 dans la fonction (matches paper)
> - Pour chaque reward `G_i` dans le batch, EMA = `α·G_i +
>   (1-α)·EMA_{i-1}` (sequential scan, scalar recurrence)
> - Label binaire : `[1, 0]` si `G > EMA` (bet, "j'ai eu mieux que
>   ma moyenne attendue"), sinon `[0, 1]` (no-bet)
> - Output shape `(B, 2)` float32
>
> **Vectorisation de label construction** : la boucle EMA reste
> sequential (parity), mais la construction des labels (`new_tensor[:,
> 0] = bet_mask`) est vectorisée — gain marginal, pas de divergence.

## 2. Contraintes scientifiques

### Mnih et al. (2015) DQN replay

Pattern canonique : cyclic buffer FIFO, uniform sampling without
replacement. Garantit que les transitions samplées sont
i.i.d.-ish (décorrellation temporelle qui casse l'auto-corrélation
de l'env step).

### `target_wager` = "did I beat my running average ?"

Le wager target en SARL est différent de Blindsight (présence binaire)
et AGL (exact reconstruction). Ici c'est **"le reward observé
dépasse-t-il l'EMA des rewards récents ?"**.

- α=0.45 (paper Table 11) → EMA reactive, regarde les 2-3 rewards
  passés
- Si `G > EMA` → "j'ai eu un meilleur reward que prédit" → bet high
- Si `G ≤ EMA` → "comme attendu ou pire" → no-bet

C'est un signal de **surprise relative** : le wager apprend à
prédire la "surprise" de la prochaine récompense. Lien conceptuel
avec les théories prédictives (Rescorla-Wagner, Friston).

## 3. Contraintes d'ingénierie

### `random.sample` vs alternatives

Garder Python `random.sample` est crucial pour parity bit-exact :
- `np.random.choice` consomme du legacy numpy state, différent
- `torch.randperm` consomme du torch state, différent
- `random.sample` consomme du Python random state, qui est seedé
  par `set_all_seeds`

### Cyclic overwrite math

```python
self.buffer[self.location] = Transition(*args)
self.location = (self.location + 1) % self.buffer_size
```

Le ordering `assign-then-increment` est important : `location` pointe
vers le slot le plus ancien à overwrite. Différent si on faisait
`increment-then-assign`.

### Sequential EMA loop

```python
for i in range(batch_size):
    ema = scaled_alpha * flattened_rewards[i] + (1 - scaled_alpha) * ema
    ema_series[i] = ema
```

C'est la boucle qui est **32% du wall sur SARL setting 1** d'après
le profilage (cf. `docs/sprints/sprint-07-profile.md`). Vectorisable
via Numba @njit (gain ~10-15% global). C'est la priorité perf #2 dans
le méta-plan.

### `target_wager` int vs float pitfall

Le student `target_wager` peut silently truncate les EMA float si on
n'attention pas. Notre port force `dtype=torch.float32` explicitement
pour éviter ce piège (les MinAtar rewards sont int).

## 4. Deviations / dettes présentes

### Pas de déviation paper sur ce module

Toutes les valeurs (buffer_size, sampling method, alpha) sont
config-driven dans `training_loop.py`. Le module data lui-même est
mécanique.

### D-sarl-alpha-ema (D.2, resolved)

L'α EMA = 0.45 (paper Table 11) — mais student shell passait 0.25
(`-ema 25`). Sprint-08 D.2 a vectorisé `target_wager` + aligné à
0.45 par défaut. Override `-o alpha=25` pour reproduire student.

Différent de l'α cascade = 0.02. Les 2 sont des "α" non-reliés. D'où
le nom `cascade_rate` dans `cascade.py` pour distinguer.

### Pas de DETTE active

## 5. Questions ouvertes

- **Q1 :** Le label `[1, 0]` vs `[0, 1]` est mathématiquement
  équivalent à un seul bit (high_wager ∈ {0, 1}). Pourquoi le double
  encoding ? Symétrie avec Blindsight (2-unit raw logits paper-faithful) ?

- **Q2 :** L'EMA est calculée dans l'ordre du batch. Si on shuffle le
  batch, les labels changent. Le student le fait dans l'ordre temporel
  (transitions samplées dans l'ordre de leur insertion ? ou shuffled ?).
  À vérifier — peut affecter parity.

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- `Transition` namedtuple field order
- Cyclic Python list buffer + `random.sample`
- `target_wager` α in percent (parity student)
- Sequential EMA loop (parity)

### À améliorer
- **Numba @njit `target_wager`** : ~10-15% gain global SARL setting 1
- **Type hint `Transition`** : utiliser `NamedTuple` typed
- **Validation `add()`** : check tuple length

### Tests à écrire
- Bit-exact parity vs reference `random.sample` (seed fixé)
- Cyclic overwrite : buffer plein, add → oldest removed
- `target_wager(rewards, 45)` produit (B, 2) avec valeurs ∈ {0, 1}
- EMA sequential consistency : 2 appels avec mêmes rewards → mêmes
  EMA

### Connexion
- Utilisé par : `domains/sarl/{training_loop, trainer}`
- Pas de config YAML dédiée
