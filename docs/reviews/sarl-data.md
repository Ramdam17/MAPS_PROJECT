# Component review — `src/maps/experiments/sarl/data.py`

**Review sub-phase :** Sprint-08 D.1 (review + map findings to planned D.2 fix).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl/data.py` (118 L, 1 namedtuple + 1 class + 2 functions).
**Paper sources :** Table 11 (SARL hyperparams), paper §3 training loop, eq.13-14 (EMA wagering).
**Student sources :** `external/paper_reference/sarl_maps.py:282-323` (replay_buffer + get_state),
L514-532 (target_wager).
**Callers du port :**
- `src/maps/experiments/sarl/trainer.py:50, 150, 157`
- `src/maps/experiments/sarl/training_loop.py:64, 239, 250, 277, 289, 303`
- `src/maps/experiments/sarl/evaluate.py:50, 147, 189`
- `src/maps/experiments/sarl_cl/trainer.py:63, 232, 239`
- `src/maps/experiments/sarl_cl/training_loop.py:55, 344, 355, 388, 399, 418`
- `tests/parity/sarl/...` (à vérifier au premier test failing)

**DoD D.1** : doc créé, 4 symbols audités, fixes tracés vers D.2, 0 code touché.

---

## `Transition` namedtuple (D.1-a)

Port L37 :
```python
Transition = namedtuple("Transition", "state, next_state, action, reward, is_terminal")
```

Student `sarl_maps.py:289` :
```python
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
```

✅ **Field order bit-exact.** Nommage `Transition` (PascalCase) vs student `transition` (lowercase)
— cosmétique, zéro impact parity. Les 2 callers du port utilisent `Transition(*zip(*sample, strict=True))`
(unpack pattern), donc positional → agnostic au nom.

## `SarlReplayBuffer` class (D.1-b)

Port L40-79 vs student `sarl_maps.py:290-307` :

| Aspect                 | Student                                                  | Port                                                          | Match |
|:-----------------------|:---------------------------------------------------------|:--------------------------------------------------------------|:-----:|
| `__init__(buffer_size)` | `self.buffer_size, self.location=0, self.buffer=[]`      | idem (type hint `list[Transition]`)                           | ✅    |
| `add(*args)` logic      | append si `len < buffer_size`, sinon overwrite `[location]`; location++ mod buffer_size | idem                                                      | ✅    |
| `sample(batch_size)`    | `random.sample(self.buffer, batch_size)`                 | idem                                                          | ✅    |
| `__len__`               | **Absent student**                                       | `return len(self.buffer)` (port ajout)                        | ⚠️ port ajout non-breaking |

✅ **Parity math bit-exact.** L'ajout `__len__` permet aux callers d'écrire
`len(buffer) >= replay_start_size` au lieu de `len(buffer.buffer) >= replay_start_size` — gain
API proprété sans changer comportement.

### RNG parity — CRITIQUE

Le docstring L6-15 documente la contrainte : `random.sample` **doit** rester le seul sampling.
Alternatives briseraient la parity :
- `np.random.choice` → consomme NumPy stream (déterministe via `np.random.seed`, mais différent
  draw que student).
- `torch.randperm` → consomme torch stream, idem.
- Tenseurs intermédiaires → changerait l'ordre d'insertion sous wrap-around.

✅ **Contrainte docstring correctement documentée, port respecte.**

## `get_state(s, device)` (D.1-c)

Port L82-90 :
```python
def get_state(s, device="cpu"):
    return torch.tensor(s, device=device).permute(2, 0, 1).unsqueeze(0).float()
```

Student `sarl_maps.py:322-323` :
```python
def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()
```

où `device` student = module-level global (`device = torch.device("cuda" if ... else "cpu")`).

**Différence unique** : port paramétrise `device=` (default CPU) vs student hardcode global.

✅ **Math bit-exact.** Gain port : testabilité (pas besoin de monkeypatch global) + explicit device
routing via `cfg.device`. Zero divergence numérique avec même device.

### Shape sanity

- Input MinAtar state : `(10, 10, C)` numpy array (H, W, C).
- `.permute(2, 0, 1)` → `(C, 10, 10)` (channels-first, PyTorch convention).
- `.unsqueeze(0)` → `(1, C, 10, 10)` (batch-first).
- `.float()` → float32.

Consommé par `SarlQNetwork.forward(x: (B, C, 10, 10))`. Cohérent shape-wise.

## `target_wager(rewards, alpha)` (D.1-d)

Port L93-118 vs student `sarl_maps.py:514-532` :

| Step                   | Student                                         | Port                                              | Match |
|:-----------------------|:------------------------------------------------|:--------------------------------------------------|:-----:|
| Flatten rewards        | `flattened_rewards = rewards.view(-1)`          | idem                                              | ✅    |
| Scale alpha (percent)  | `alpha = float(alpha/100)`                       | `scaled_alpha = float(alpha / 100)` (renamed)    | ✅    |
| EMA init               | `EMA = 0.0`                                      | `ema = 0.0`                                       | ✅    |
| Output tensor          | `torch.zeros(batch_size, 2, device=rewards.device)` | idem                                           | ✅    |
| Loop update            | `EMA = alpha * G + (1 - alpha) * EMA`            | `ema = scaled_alpha * g + (1 - scaled_alpha) * ema` | ✅  |
| Labelling              | if `G > EMA`: `[1, 0]` else `[0, 1]`             | idem                                              | ✅    |
| Return                 | `new_tensor` shape `(batch, 2)`                  | idem                                              | ✅    |

✅ **Math bit-exact avec student.**

### 🆘🆘 D-sarl-alpha-ema — config divergence (déjà trackée)

**Paper Table 11 / eq.13-14** : `α_EMA = 0.45`.
**Student shell default** : `-ema 25` → `alpha/100 = 0.25` (paper-divergent).
**Port config `sarl.yaml:44`** : `alpha: 1.0` → `alpha/100 = 0.01` (paper-divergent ET student-divergent).

Le code `target_wager` est **correct** — c'est la **valeur config** qui diverge. Le fix est déjà
queued en **Phase D.2** (plan ligne 494) : *"Fix target_wager vectorisation + alpha alignment"*.

**Action D.1 : aucune**. Flag pour D.2.

### (e) Performance — vectorisation possible ?

Port L111-117 est un Python `for` loop O(B) séquentiel. Pour `batch_size=32` (paper), ~30 iterations
triviales → pas de bottleneck pratique.

**Vectorisation théorique** : la récurrence `EMA_i = α·G_i + (1-α)·EMA_{i-1}` est une filtre IIR
ordre 1 → formule fermée :

$$EMA_i = \sum_{k=0}^{i} \alpha (1-\alpha)^{i-k} \cdot G_k$$

Implémentable en O(B) vectorisé via `torch.cumsum(G * decay_factors) / normaliser` ou via
`torchaudio.functional.lfilter` (coef `b=[α]`, `a=[1, -(1-α)]`).

**Piste D.1-perf-1** (skip D.1, prévu D.2) : vectoriser pour batches plus grands + numerical
stability. **Reste bit-parity** seulement si les opérations float sont équivalentes — à vérifier
avec `torch.allclose(vectorized, loop, atol=1e-7)`.

**Action D.1 : aucune**. Flag pour D.2.

### (f) `rewards.size(0)` vs `flattened_rewards.size(0)`

Port L109 : `batch_size = rewards.size(0)`. Si `rewards.shape = (B,)`, OK. Si `(B, 1)` ou `(B, k)`
avec `k > 1`, on prend `B` (la 1ère dim) et on flatten à `(B * k,)` derrière → `flattened_rewards[i]`
indexe jusqu'à `B-1`, donc on **n'utilise que les `B` premiers éléments** de la flatten — pas
`B * k`. Si caller passe `rewards.shape = (B, 2)`, la moitié est ignorée silencieusement.

**Student** : **même comportement**, même piège.

**Action D.1 : aucune**. Flag docstring → **D1-fix-1** : docstring explicite "rewards must be
shape `(batch,)` or `(batch, 1)` ; multi-dim shapes are silently truncated".

### Callers audit

| Caller                                         | Usage                                              | Status |
|:-----------------------------------------------|:---------------------------------------------------|:------:|
| `sarl/training_loop.py:303`                    | `target_wager_fn=target_wager` passé au trainer    | ✅    |
| `sarl/trainer.py:157`                          | `targets_wagering = target_wager_fn(rewards, alpha)` | ✅  |
| `sarl_cl/training_loop.py:418` + `sarl_cl/trainer.py:239` | idem                                          | ✅    |
| `sarl/rollout.py:118`                          | `get_state` passé via helper                       | ✅    |
| `sarl/evaluate.py:147, 189`                    | `get_state` direct                                 | ✅    |
| `sarl/training_loop.py:239`                    | `SarlReplayBuffer(cfg.replay_buffer_size)`         | ✅    |
| `sarl/training_loop.py:289`                    | `buffer.sample(cfg.batch_size)`                    | ✅    |

Tous les callers correct.

## Fixes identifiées D.1

| ID       | Fix                                                                   | Scope                           | Effort |
|:---------|-----------------------------------------------------------------------|---------------------------------|:------:|
| D1-fix-1 | Docstring `target_wager` : préciser shape contrainte `(batch,)` ou `(batch, 1)` | `data.py:93-105` docstring | 3 min  |
| D1→D.2   | Vectoriser `target_wager` loop (perf)                                 | `data.py:93-118` impl           | queued D.2 |
| D1→D.2   | Aligner `config/training/sarl.yaml:44 alpha=1.0` → `45` (paper Table 11) | `config/training/sarl.yaml:44` | queued D.2 |

## Cross-reference deviations.md

- **D-sarl-alpha-ema** (🆘🆘 double) déjà tracée. Ré-confirmée par cette review : code correct,
  config en divergence. Fix queued D.2.
- **Aucune nouvelle deviation** surfacée.

## Résumé — `sarl/data.py`

- ✅ **`Transition` field order** bit-exact.
- ✅ **`SarlReplayBuffer`** math + RNG parity student (ajout `__len__` safe).
- ✅ **`get_state`** math bit-exact (seule diff : `device` param vs global).
- ✅ **`target_wager`** math bit-exact avec student ; loop séquentiel O(B) non-optim.
- 🆘🆘 **D-sarl-alpha-ema** : code correct, config diverge paper + student — fix D.2.
- ⚠️ **1 piège silencieux** : `rewards.shape` multi-dim → truncation silencieuse → docstring fix.
- **0 divergence code paper-vs-port, 1 divergence config (déjà queued), 1 fix docstring.**

**D.1 clôturée. 1 fix docstring pour D.2 batch (ou intégré au D.2 fix alpha). 2 fixes queued D.2
confirmés.**
