# Component review — `src/maps/experiments/sarl/rollout.py`

**Review sub-phase :** Sprint-08 D.5.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl/rollout.py` (209 L, 1 dataclass + 3 functions).
**Paper sources :** §3.2 (ε-greedy schedule), Table 11 (EPSILON_START=1.0, END_EPSILON=0.1,
FIRST_N_FRAMES=100000, REPLAY_START_SIZE=5000).
**Student sources :** `external/paper_reference/sarl_maps.py:462-495` (`world_dynamics`) + L95-105
(module-level constants).
**Callers du port :**
- `sarl/training_loop.py:289+` — rollout step (training).
- `sarl/evaluate.py:~147+` — greedy rollout (evaluation).

**DoD D.5** : doc créé, 4 symbols audités, cascade no-op cross-ref D.4, 0 code touché.

---

## Module organization — pure-function split

Le port **découpe** le `world_dynamics` student en 2 morceaux :
- Action selection (pure, testable sans env) → `rollout.py`.
- Env stepping (`env.act`, `env.state`) → reste dans `training_loop.run_training`.

Docstring L8-20 justifie : student `world_dynamics` fait 2 jobs en 1 appel, le test demande un env
MinAtar live sur le bench → couplage indésirable. Split port = **gain testabilité**, math
préservée.

✅ **Split architectural sûr, paper-math préservée.**

## Module-level constants (D.5-a)

```python
EPSILON_START = 1.0
END_EPSILON = 0.1
FIRST_N_FRAMES = 100_000
```

Vs student `sarl_maps.py:95-105` :
```python
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000  # (utilisé via cfg dans le port)
END_EPSILON = 0.1
EPSILON = 1.0             # renamed EPSILON_START dans le port pour clarté
```

Vs paper Table 11 (déjà extraite dans `paper_tables_extracted.md`) :
- ε_start = 1.0, ε_end = 0.1, annealing frames = 100,000, replay_start = 5,000.

✅ **4 constants bit-exact paper + student.** Renaming cosmétique `EPSILON` → `EPSILON_START`
sans impact.

**⚠️ Hardcodées** en module-level (pas config-driven). Paper values sont stables sur les 5 games
→ OK en pratique. Si un futur scénario (ex: jouer avec ε_start pour ablation) demande override,
il faudrait passer par config. **Non-bloquant**.

**Piste D5-fix-1** (debate, skip) : expose-les via `SarlTrainingConfig` + default paper values.
Marginal gain, skip.

## `anneal_epsilon(t, replay_start_size)` (D.5-b)

Port L72-97 vs student L474-475 :

**Student (1-liner branchless)** :
```python
epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
    else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON
```

**Port (3-branche lisible)** :
```python
if t < replay_start_size:
    return EPSILON_START  # warmup
progress_frames = t - replay_start_size
if progress_frames >= FIRST_N_FRAMES:
    return END_EPSILON
slope = (END_EPSILON - EPSILON_START) / FIRST_N_FRAMES
return EPSILON_START + slope * progress_frames
```

**Différence visible** : le port ajoute une **branche explicite warmup** (L89-90) qui retourne
`EPSILON_START`. Le student ne la traite **pas dans anneal** — elle est gérée par le `if t <
replay_start_size` dans `world_dynamics` L468 **avant** même d'appeler le calcul ε.

**Impact** :
- Paper / student : pendant warmup, ε n'est **jamais calculée** (code mort avant anneal).
- Port : pendant warmup, `anneal_epsilon(t < replay_start, ...)` retourne `EPSILON_START=1.0`,
  mais le caller (`epsilon_greedy_action` L147-154) **court-circuite** également l'appel d'ε avant
  d'y arriver → la valeur retournée est utilisée **uniquement pour logging** (`ActionSelection.epsilon`).
  Docstring L85-87 documente cette intention.

✅ **Math parity.** Valeurs ε identiques à chaque `t ≥ replay_start`. Return warmup = convention
interne port, pas de divergence.

## `epsilon_greedy_action(...)` (D.5-c)

Port L100-188 vs student L462-495 (action-selection slice) :

| Step                       | Student                                         | Port                                                       | Match |
|:---------------------------|:------------------------------------------------|:-----------------------------------------------------------|:-----:|
| Warmup check               | `if t < replay_start_size:` L468               | idem L147                                                  | ✅    |
| Random action warmup       | `torch.tensor([[random.randrange(num_actions)]])` | `torch.tensor([[(python_rng or random).randrange(...)]])` | ✅ same RNG stream |
| ε draw                     | `numpy.random.binomial(1, epsilon) == 1`        | `numpy.random.binomial(1, epsilon)` (via `numpy_rng` or legacy) | ✅ |
| Explore (ε path)           | `random.randrange(num_actions)` L478           | idem L164                                                  | ✅    |
| Greedy branch cascade loop | `for j in range(cascade_iterations_1): ... policy_net(s, main_task_out, cascade_rate_1)` L483-485 | idem L176-178 | ✅ |
| Argmax                     | `output_network_policy.max(1)[1].view(1, 1)`   | `q_values.max(1)[1].view(1, 1)`                            | ✅    |

✅ **Bit-parity step-by-step avec student.**

### RNG parity — 3 streams critiques

Le port préserve les **3 streams** dans le **même ordre** que student :
1. **`random.randrange`** (Python stdlib) — warmup + explore action index.
2. **`numpy.random.binomial(1, ε)`** (legacy NumPy global) — explore-vs-exploit coin flip.
3. **`torch` forward** — greedy cascade (no RNG dans un `torch.no_grad()` block avec
   `SarlQNetwork.forward` déterministe).

**Ordre d'appel par step** :
- Si warmup : stream 1 (1 draw).
- Sinon : stream 2 (1 draw) → si explore, stream 1 (1 draw) ; si exploit, stream 3 (0 RNG).

→ **Même consommation RNG que student, même séquence de valeurs avec même seed**. Critical pour
parity tests.

**⚠️ `numpy.random.binomial` legacy** : le docstring L159-161 confirme que port utilise le global
numpy legacy (pas `np.random.default_rng()`) pour match student. `noqa: NPY002` ajouté. Cohérent
avec finding C13-F1 (seeding covers legacy, not default_rng). ✅

### `python_rng` / `numpy_rng` injection (port addition)

Port ajoute `python_rng: random.Random | None = None` et `numpy_rng: numpy.random.Generator |
None = None` pour **tests déterministes**. Student n'a pas ce paramètre.

**Gain** : possibilité de créer un `random.Random(seed=42)` local pour isoler un test ; par
défaut tombe sur les globals (= student behavior exact).

✅ **Port addition non-breaking, aucun impact production.**

### 🚨 D.4 cascade no-op re-emphasized — **rollout path is the real compute cost**

Le cascade loop L176-178 s'exécute sur **CHAQUE env step** (pas seulement update steps). Sur
5M frames avec `cascade_iterations_1=50` :
- 5,000,000 env steps × 50 forwards = **250,000,000 SarlQNetwork.forward calls**.
- Toutes sont mathématiquement identiques par step (C.1 no-op).
- → **99% du compute forward est gaspillé**.

**Update vs rollout** : D.4 ajoutait le log+metadata pour le compute waste sur le path **update**
(~ num_frames / training_freq updates = 5M/4 = 1.25M updates × 50 = 62.5M). Le path rollout
(actions) est **4× plus coûteux** (250M vs 62.5M forwards gaspillés).

**Action** : le log `log.warning` au setup de `run_training` (D.4) couvre déjà le cas globalement.
Pas d'action spéciale en D.5 — le comment inline dans `SarlQNetwork.forward` (D.4) s'applique
aux deux paths (rollout + update).

Si Option B (shortcut) devait être adoptée en Phase H, elle ferait **ici** son gain maximal.
Flag pour mémoire.

## `greedy_action(...)` (D.5-d)

Port L191-209 — pure greedy pour evaluation :

```python
def greedy_action(state, policy_net, cascade_iterations_1):
    cascade_rate_1 = 1.0 / cascade_iterations_1
    main_task_out = None
    with torch.no_grad():
        for _ in range(cascade_iterations_1):
            q_values, _, _, main_task_out = policy_net(state, main_task_out, cascade_rate_1)
    return q_values.max(1)[1].view(1, 1)
```

Idem cascade loop comme greedy branch de `epsilon_greedy_action`. Split par intent (eval vs
training rollout).

✅ **Pas de divergence, juste un alias clair pour evaluate.py.**

**Piste D5-fix-2** (skip) : factoriser le cascade forward en helper interne
`_cascade_forward(policy_net, state, iters)` utilisé par epsilon_greedy_action ET greedy_action.
Marginal DRY gain, skip.

## `ActionSelection` dataclass — port addition

Port L58-69, non-présent student :

```python
@dataclass
class ActionSelection:
    action: torch.Tensor        # (1, 1) int64
    epsilon: float              # NaN during warmup
    was_exploration: bool
    q_values: torch.Tensor | None = None
```

Utilisé par le caller (`training_loop`) pour :
- Action à passer à `env.act()`.
- Stats logging (ε curve, exploration fraction).
- Debug (q_values pour monitoring greedy predictions).

✅ **Gain port pur** (debugabilité) sans impact math/RNG. Aucun caller student cassé (student
retourne tuple dans `world_dynamics` mais le port split lui-même le remplace).

## Callers audit

| Caller                                | Usage                                           | Status |
|:--------------------------------------|:------------------------------------------------|:------:|
| `sarl/training_loop.py:289`           | `epsilon_greedy_action(state, policy_net, t, ...)` | ✅  |
| `sarl/evaluate.py:147, 189`           | `greedy_action(state, policy_net, cascade_iters_1)` | ✅ |

## Fixes identifiées D.5

| ID       | Fix                                                                      | Scope                           | Effort |
|:---------|--------------------------------------------------------------------------|---------------------------------|:------:|
| D5-fix-1 | (skip) expose EPSILON_START/END_EPSILON/FIRST_N_FRAMES via SarlTrainingConfig | — | skip (paper values stable) |
| D5-fix-2 | (skip) factoriser cascade forward helper                                 | —                               | skip (DRY marginal) |

## Cross-reference deviations.md / reviews

- **D.4 cascade no-op re-confirmed** — cost = 250M wasted forwards en rollout (vs 62.5M en update).
  Option A (paper-faithful keep) déjà décidée. Post-repro Option B optim = rollout.py ciblé.
- **C13-F1 `numpy.random.default_rng` gap** : rollout.py utilise `numpy.random.binomial` legacy
  global, couvert par `set_all_seeds`. Coherent avec le gap documenté.
- **Aucune nouvelle deviation** surfacée.

## Résumé — `sarl/rollout.py`

- ✅ **Architectural split** (action vs env stepping) paper-math préservée.
- ✅ **4 module constants** bit-exact paper + student.
- ✅ **`anneal_epsilon`** math parity (port ajoute branche warmup explicite pour logging, sans
  impact comportement).
- ✅ **`epsilon_greedy_action`** bit-parity 6 steps.
- ✅ **RNG parity** 3 streams (Python random, NumPy legacy binomial, torch forward) dans le
  même ordre que student.
- ✅ **`ActionSelection` dataclass** — gain port pure (debug+test).
- ✅ **`greedy_action`** = alias clair pour eval (pas de divergence).
- ⚠️ **D.4 cascade no-op re-confirmed** — le rollout path est le **vrai hotspot** de compute
  waste (250M forwards gâchés). Déjà tracé par D.4 logging + metrics JSON, rien à faire ici.
- **0 divergence paper-vs-port, 0 fix bloquant, 2 skips hors scope.**

**D.5 clôturée. 0 code touché.**
