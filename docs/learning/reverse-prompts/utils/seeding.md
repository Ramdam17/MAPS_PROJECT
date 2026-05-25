# Reverse-prompt — `src/maps/utils/seeding.py`

**Module path actuel (sur main) :** `src/maps/utils/seeding.py`
**Taille :** ~75 lignes, 1 fonction publique (`set_all_seeds`).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `seeding.py` qui seed **toutes** les sources
> de stochasticité utilisables par un run MAPS. CLAUDE.md exige seed
> control "à chaque entry point" — ce module est le helper unique
> appelé partout.
>
> API publique : `set_all_seeds(seed, *, deterministic_cudnn=True)`.
>
> Source à seed :
> 1. Python `random.seed(seed)`
> 2. NumPy **legacy global API** `np.random.seed(seed)` — le student
>    utilise `np.random.normal/randint/...` partout
> 3. PyTorch `torch.manual_seed(seed)` (couvre CPU + MPS — Apple
>    Silicon hérite du global torch seed)
> 4. PyTorch CUDA `torch.cuda.manual_seed_all(seed)` (si dispo)
> 5. cuDNN deterministic mode (si CUDA dispo et `deterministic_cudnn=True`)
> 6. `PYTHONHASHSEED` env var (affecte iteration order de certains dicts)
>
> **Ne PAS** activer `torch.use_deterministic_algorithms(True)` —
> raise sur des ops non-déterministes légitimes (scatter_add CUDA en
> RL). À activer per-domain si full determinism requis.
>
> Gap connu : `np.random.default_rng()` (PCG64) n'est PAS seedé par
> `np.random.seed`. Document explicit. Aucun code actuel ne l'utilise.
>
> Log debug du seed appliqué pour traçabilité.

## 2. Contraintes scientifiques

### Reproductibilité = mandatory

CLAUDE.md : "Seed control is mandatory — `maps.utils.seeding.set_all_seeds(seed)`
at every entry point". Les Tables paper sont reproduites à seed fixé.
La convention lab est **seed=42**.

### Pourquoi `numpy.random.seed` (legacy) et pas `default_rng` ?

Le code student utilise `np.random.normal()`, `np.random.randint()`,
etc. qui passent par le **legacy global state**. `np.random.seed`
configure ce state. Si on portait vers `default_rng()` (qui est
préférable depuis NumPy 1.17), chaque site d'appel aurait son propre
Generator à seedier indépendamment — refactor non-trivial.

→ Garder legacy par parity. Migration future = refactor parallèle.

## 3. Contraintes d'ingénierie

### Pourquoi `deterministic_cudnn=True` par défaut ?

cuDNN sélectionne automatiquement l'algo conv le plus rapide selon
le warmup. Cet auto-tuning est non-déterministe d'un run à l'autre. En
mode deterministic, cuDNN se fige sur un algo (slower mais stable).

Cost : throughput perdu (~5-15% sur GPU). Bénéfice : reproductibilité
seed-pinned. Pour MAPS où on compare des Tables à 3 décimales, le
bénéfice domine.

### Pourquoi pas `use_deterministic_algorithms(True)` ?

C'est la version "nuclear" — raise sur **toute** op non-déterministe
(scatter_add CUDA, atomic ops, etc.). RL code (SARL) utilise ces ops
légitimement (replay buffer indexing, value updates). Activer ça
crasherait le run. Donc : pas activé par défaut, à activer
domain-specific si besoin.

### Edge cases

- **MPS (Apple Silicon)** : `torch.manual_seed` couvre MPS depuis
  PyTorch ≥2.0. Aucun API séparé.
- **Multiprocess** : seed le master seulement ; les workers doivent
  re-seed (DataLoader avec `worker_init_fn=lambda i: set_all_seeds(seed
  + i)`).
- **Bursts of randomness avant seed** : si on import des modules qui
  appellent np.random à l'import, on est foutu. CLAUDE.md dit "before
  importing data, building models" → l'appel doit être dans la première
  ligne d'entry point.

## 4. Deviations / dettes présentes

Pas de déviation paper.

**Pas de dette active.** Module est bien fait. Une seule chose à
documenter explicitement : si on migre vers `default_rng()`, on doit
re-architecturer (chaque consommateur récupère le Generator au lieu de
`np.random.<func>`).

## 5. Questions ouvertes

- **Q1 :** `PYTHONHASHSEED` est seedé via `os.environ` — mais Python
  ne re-read pas cette env var après le démarrage de l'interpreter.
  Pour que `PYTHONHASHSEED` fonctionne, il **doit être set AVANT le
  démarrage**. Notre `set_all_seeds` est appelée APRÈS l'import donc
  c'est **trop tard**. Bug subtil — le seed n'a aucun effet sur le
  hash() de cette session. À investiguer.

## 6. Pour le rewrite (Sprint 11+)

### À garder
- API `set_all_seeds(seed, *, deterministic_cudnn=True)`
- Tous les 6 sources couvertes
- Log debug du seed
- Comportement "appeler partout en entry point"

### À améliorer
- **Q1 PYTHONHASHSEED bug** : soit retirer cette ligne (futile), soit
  documenter que la convention est `PYTHONHASHSEED=42 uv run script.py`
- **Worker seed helper** : `make_worker_init_fn(base_seed)` pour
  DataLoader workers
- **CUDA fork detection** : si la process forke (multiprocess RL),
  re-seed dans le child

### Tests à écrire
- Appel `set_all_seeds(42)` → `random.random()` produit valeur
  déterministe
- Idem pour `np.random.rand()`
- Idem pour `torch.randn(10)`
- cuDNN deterministic mode actif si CUDA dispo
- Double appel `set_all_seeds(42)` produit même output (idempotence)

### Connexion
- Appelé en tête de tous les `cli.py` (sprint 11+)
- Appelé dans les tests qui ont besoin de reproductibilité

## Méta — pour Claude chat

```text
Je veux comprendre les pièges de la reproductibilité numérique en
PyTorch quand on utilise multiple sources de stochasticité (Python
random, NumPy legacy global, NumPy default_rng, torch CPU/CUDA/MPS,
cuDNN autotuner).

1. Quelles sont les sources cachées que j'oublie souvent (data
   augmentation libraries, scikit-learn estimators random_state) ?
2. Pour des comparaisons rigoureuses (paper reproduction à 3 décimales),
   est-ce que cudnn.deterministic=True suffit, ou faut-il aussi
   use_deterministic_algorithms(True) avec ses cascading failures ?
3. Comment seed proprement dans un setup DataLoader num_workers>0 ?
```
