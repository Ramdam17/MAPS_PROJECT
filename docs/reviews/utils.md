# Component review — `src/maps/utils/` (minus `energy_tracker.py`)

**Review sub-phase :** Sprint-08 C.13 (5 modules reproduction-critiques : `seeding.py`, `config.py`,
`paths.py`, `logging_setup.py`, `__init__.py`).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**Files under review :**
- `src/maps/utils/seeding.py` (60 L)
- `src/maps/utils/config.py` (158 L)
- `src/maps/utils/paths.py` (106 L)
- `src/maps/utils/logging_setup.py` (71 L)
- `src/maps/utils/__init__.py` (15 L)

**Skipped (→ C.14):** `src/maps/utils/energy_tracker.py` (612 L, non-reproduction-critical monitoring).

**DoD global C.13** : doc créé, 5 sous-sections, fixes identifiées, 0 code touché.

---

## `seeding.py` (C.13-a)

### Port complet (30 lignes utiles)

```python
def set_all_seeds(seed: int, *, deterministic_cudnn: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
```

### (a) Couverture RNG

| RNG                                 | Couvert ? | Note |
|:------------------------------------|:---------:|:-----|
| `os.environ['PYTHONHASHSEED']`      | ✅        | affecte iter dict dans certains paths (CPython ≥ 3.3) |
| `random` (stdlib)                   | ✅        |      |
| `numpy.random`                      | ✅        | couvre aussi `np.random.default_rng`? → NON, seulement legacy singleton |
| `torch` CPU                         | ✅        | `torch.manual_seed` seed aussi MPS via backend torch |
| `torch.cuda` (si dispo)             | ✅        | `manual_seed_all` pour multi-GPU |
| `torch.backends.cudnn.deterministic`| ✅        | forcé True (sauf si caller opt-out) |
| `torch.backends.cudnn.benchmark`    | ✅        | forcé False (auto-tuning désactivé) |
| `torch.use_deterministic_algorithms`| ❌ **doc** | raise sur ops RL (scatter_add CUDA), opt-in par domaine |
| MinAtar `env.seed()`                | ❌ **implicite** | MinAtar utilise `np.random` globalement → couvert indirectement par `np.random.seed` |

**Finding C13-F1** (info) : `np.random.default_rng()` — le nouveau API NumPy (PCG64) — n'est **pas**
seed par `np.random.seed`. Si un futur module utilise `default_rng()` il faudra un seed explicite.
Le port actuel (Blindsight/AGL/SARL) utilise le **legacy API** (`np.random.normal`, `np.random.randint`)
donc OK pour la reproduction paper.

**Piste C13-fix-1** : docstring `seeding.py` ajouter une note "This seeds the **legacy** NumPy RNG
singleton. Code using `np.random.default_rng()` must seed independently." Zero risk actuel, mais
défense future.

### (b) Parity student

**Student Blindsight** (`blindsight_tmlr.py:1196`) :
```python
temperature = 1.0
# ... plus loin, pas d'appel à random.seed ni np.random.seed(...)
```

**Student AGL + SARL** : même pattern, **aucun appel explicite à set_all_seeds equivalent**. Le
student repose sur l'ordre d'import et la JVM-like séquence Python comme source de déterminisme
— **non-déterministe par défaut** en pratique.

**Port** améliore sur student : seeding global via `set_all_seeds`. Paper-faithful requiert
reproduction bit-exact, donc port ≥ student.

✅ **Port = amélioration sur student, sans divergence numérique.**

### (c) MinAtar env seed responsibility

`MinAtar.Environment` utilise `np.random` pour `reset`/`action` stochastique. Seedée indirectement
via `np.random.seed(seed)` de `set_all_seeds`. **Pas d'appel explicite `env.seed()`** dans les
trainers SARL — OK tant que MinAtar utilise le singleton legacy.

**Piste C13-fix-2** (info) : flagger dans `sarl/training_loop.py:177` que `torch.manual_seed(cfg.seed)`
direct ≠ `set_all_seeds(cfg.seed)`. Actuellement OK car script CLI (`run_sarl.py:184`) appelle
`set_all_seeds` avant, mais fragile si `run_training` est appelé programmatiquement sans wrapper.

### (d) Résumé `seeding.py`

- ✅ Couverture RNG quasi-complète (Python, NumPy legacy, torch CPU/CUDA/MPS, cudnn).
- ⚠️ Gap documenté : `torch.use_deterministic_algorithms`, `np.random.default_rng`.
- ✅ Amélioration nette sur student (qui ne seed rien).
- **2 fixes docstring** pour C.14 batch (C13-fix-1, C13-fix-2).

---

## `config.py` (C.13-b)

Port (158 L) : `load_config(name, overrides, resolve)` + `_apply_defaults` (Hydra-lite).

### (a) Architecture

- `_find_project_root()` walks up depuis `cwd` pour trouver `config/maps.yaml`.
- `_resolve_config_path(name)` accepte 3 formes : bare (`"maps"`), sub-path (`"training/blindsight"`),
  absolute path.
- `_apply_defaults(cfg, path)` : résoud `defaults:` list Hydra-style, récursif.
- `load_config(name, overrides, resolve)` : entry point, applique overrides via `OmegaConf.from_dotlist`.

### (b) Couverture Hydra — sous-ensemble minimal

Support :
- `- /maps@_here_` (merge `config/maps.yaml` à root).
- `- training/foo` (merge `config/training/foo.yaml` à root).
- `_self_` sentinel (toléré, ignoré — Hydra convention).

**Non-support** (documenté implicitement) :
- Overlays `@group` (ex: `- config/optimizer@model.optim`).
- Runtime composition via `--config-path` CLI.
- Plugins / launchers.

✅ **Minimaliste mais couvre le besoin MAPS.**

### (c) 🚨 Finding C13-F2 — `CONFIG_ROOT` résolu au module-import time

```python
CONFIG_ROOT: Path = _find_project_root() / "config"
```

C'est évalué **lors de l'import** de `config.py`, pas lors de `load_config()`. Si le `cwd` change
entre l'import et l'appel, `CONFIG_ROOT` pointe vers un mauvais root.

**Scénarios problématiques** :
1. Test avec `monkeypatch.chdir(tmp_path)` puis import — évité par fixture `autouse` early.
2. Notebook qui fait `os.chdir(...)` après import — improbable mais possible.

**Impact port actuel** : faible. La découverte via `parent containing config/maps.yaml` est robuste
sur `cwd=project_root`, qui est la norme (CLI `uv run` depuis root).

**Piste C13-fix-3** : deux options :
- (a) Lazy-resolve : faire `CONFIG_ROOT` une property/fonction appelée à chaque `load_config`.
- (b) Docstring warning : "CONFIG_ROOT resolved at import time; if cwd changes post-import, pass
  absolute path to load_config instead".

Recommandation : **(b) docstring warning**, coût minimal, cas rare.

### (d) Validation absente

`load_config` retourne un `DictConfig` opaque. Aucune validation de schema, types, required keys.
Les callers font `int(cfg.training.batch_size)` en runtime — une faute de frappe dans le YAML
surface tard (au `int(...)` cast), pas au load.

**Pattern alternatif** (Hydra propose) : `@dataclass` validators via `ConfigStore.instance().store()`.
Overkill pour MAPS actuel, mais pourrait aider pour Phase F.

**Piste C13-fix-4** (debate, skip C.13) : introduction de dataclass schemas pour les configs
training. Hors scope C.13.

### (e) Résumé `config.py`

- ✅ Hydra-lite clean, récursif, testé.
- ⚠️ `CONFIG_ROOT` import-time — docstring warning suffisant.
- ⚠️ Pas de schema validation — accepté pour parity student (aussi sans validation).
- **1 fix docstring + 1 debate skip.**

---

## `paths.py` (C.13-c)

Port (106 L) : dataclass `Paths` + `get_paths(root=None)` + `_discover_root()`.

### (a) Architecture

- `Paths` dataclass frozen avec 8 fields (root, data, outputs, models, logs, figures, reports, scratch_root).
- `_owned` tuple : exclut `root` + `scratch_root` de `ensure_dirs` (ne pas mkdir $SCRATCH qui
  pré-existe sur HPC).
- `_discover_root()` : `MAPS_ROOT` env > parent containing `config/paths.yaml` > cwd.
- `get_paths(root=None)` : charge `config/paths.yaml`, substitue `${...}` interpolations, override
  `scratch_root` avec `$SCRATCH` env si set.

### (b) $SCRATCH discipline (DRAC)

```python
scratch_env = os.environ.get("SCRATCH")
if scratch_env:
    kwargs["scratch_root"] = Path(scratch_env).resolve()
```

Sur tamia cluster : `$SCRATCH` set par défaut → `scratch_root = /scratch/$USER/`. Sur dev :
fallback `outputs/` (project-local). ✅ **Paper-faithful et CLAUDE.md-compliant** (scratch wins
on HPC, discipline Rémy mémoire).

### (c) `_substitute` — handling minimal des interpolations

```python
def _substitute(value: str) -> Path:
    if value.startswith("${"):
        _, _, tail = value.partition("}/")
        return (resolved_root / tail).resolve() if tail else resolved_root
    p = Path(value)
    return p.resolve() if p.is_absolute() else (resolved_root / p).resolve()
```

**Ne gère que** `${anything}/sub-path` ou raw path. Si `paths.yaml` utilise `${project_root}/...`
ou une interpolation OmegaConf native (ex: `${oc.env:HOME}`), le port ne les résoud pas.

Actuel `config/paths.yaml` : **aucune interpolation**, juste des chemins relatifs (`data`,
`outputs`, etc.). → Port OK en état.

**Piste C13-fix-5** (info) : `_substitute` pourrait utiliser `OmegaConf.resolve(cfg)` directement
pour tirer parti du resolver OmegaConf natif (plus robuste). Mais cela demanderait d'exposer
`root` comme env var ou pre-populated key. Refacto Phase H, pas critique.

### (d) `ensure_dirs` safety

```python
def ensure_dirs(self) -> None:
    for name in self._owned:
        getattr(self, name).mkdir(parents=True, exist_ok=True)
```

N'appelle **jamais** `root.mkdir` (project root doit exister). Ne crée pas `scratch_root`
(HPC-owned). Crée les 6 dirs `_owned`. ✅ **Safe.**

### (e) Résumé `paths.py`

- ✅ Architecture propre (dataclass frozen, auto-discovery root, $SCRATCH wins).
- ✅ DRAC-faithful (respect `$SCRATCH` env).
- ⚠️ `_substitute` minimaliste — accepte si pas d'interpolation OmegaConf native dans YAML.
- **0 fix urgent, 1 note refacto Phase H.**

---

## `logging_setup.py` (C.13-d)

Port (71 L) : `configure_logging(level, log_file, force)`.

### (a) Architecture

- Format : `%(asctime)s %(name)s [%(levelname)s] %(message)s` — standard lab.
- `StreamHandler(stderr)` toujours + `FileHandler(log_file, mode="a")` optionnel.
- `force=True` : clears handlers existants (bon pour notebooks ré-exécutés).
- Log level : string (`"INFO"`) ou int — flexible.

### (b) Lab policy compliance

**CLAUDE.md** : *"Log with Python `logging` module (not `print`) in production code."*
**Feedback Rémy** : *"Never silence console output. Do not add `quiet`, `--silent`,
`show_progress=False`, `disable=True` on progress bars, or suppress logs/prints."*

**Port** : ✅
- Pas de `quiet` / `silent` flag.
- Pas de suppression third-party logs (torch, omegaconf).
- `configure_logging` expose juste un format + level + optional file tee.

### (c) ⚠️ Pas de rotation fichier

`FileHandler(log_file, mode="a")` append-only, aucune taille max. Sur long run (SARL 5M frames),
le log peut atteindre plusieurs Go. Pas bloquant, mais piège en storage `$SCRATCH`.

**Piste C13-fix-6** (debate, skip C.13) : envisager `RotatingFileHandler(log_file, maxBytes=100e6,
backupCount=5)` pour caps à 500 Mo. Hors scope C.13 — valider avec Rémy si problème réel.

### (d) Résumé `logging_setup.py`

- ✅ Conforme CLAUDE.md (pas de silence).
- ✅ Format standardisé.
- ⚠️ Pas de rotation fichier — accepté pour runs courts, à revoir si long runs.
- **0 fix C.13, 1 debate hors scope.**

---

## `__init__.py` (C.13-e)

```python
from maps.utils.config import CONFIG_ROOT, load_config
from maps.utils.logging_setup import configure_logging
from maps.utils.paths import Paths, get_paths
from maps.utils.seeding import set_all_seeds

__all__ = ["CONFIG_ROOT", "Paths", "configure_logging", "get_paths", "load_config", "set_all_seeds"]
```

✅ **Re-exports propres, `__all__` complet et cohérent.**

---

## Fixes identifiées C.13

| ID        | Fix                                                                     | Scope                                       | Effort |
|:----------|-------------------------------------------------------------------------|---------------------------------------------|:------:|
| C13-fix-1 | Docstring `set_all_seeds` : note "legacy NumPy RNG seeded, `np.random.default_rng()` needs independent seeding" | `seeding.py:1-13` docstring | 3 min |
| C13-fix-2 | (info) Docstring `sarl/training_loop.py:177` + `sarl_cl/training_loop.py:238` — préférer `set_all_seeds` | `training_loop.py` commentaire | 5 min |
| C13-fix-3 | Docstring `config.py` : warning "CONFIG_ROOT resolved at import time"   | `config.py:40-60` docstring                 | 3 min  |
| C13-fix-4 | (skip C.13) schema validation via `@dataclass` config validators        | —                                           | skip   |
| C13-fix-5 | (skip C.13) `paths.py _substitute` refacto vers OmegaConf native        | —                                           | skip   |
| C13-fix-6 | (skip C.13) `logging_setup.py` rotation fichier                         | —                                           | skip   |

### Cross-reference deviations.md

- **Aucune nouvelle deviation** pour utils/. Les utils sont des additions port pur (student n'a
  pas d'équivalent modulaire), pas des divergences paper.

### Résumé global C.13 (5 modules)

- ✅ **5 modules propres**, 0 bug critique, architecture cohérente.
- ✅ **Conforme policies** : CLAUDE.md (logging sans silence), Rémy memory ($SCRATCH discipline).
- ⚠️ **3 gaps docstring** mineurs (C13-fix-1/-2/-3).
- **3 refactos hors scope** (-fix-4/-5/-6) post Phase F.
- **0 divergence paper-vs-port.**

**C.13 clôturée. 3 fixes docstring pour C.14/C.15 batch. Pas de code changed. Next : C.14 =
energy_tracker.py (612 L, standalone review).**
