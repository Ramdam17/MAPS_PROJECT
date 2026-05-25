# Reverse-prompt — `src/maps/utils/logging_setup.py`

**Module path actuel (sur main) :** `src/maps/utils/logging_setup.py`
**Taille :** ~60 lignes, 1 fonction publique (`configure_logging`).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `logging_setup.py` qui centralise la config
> du logging stdlib pour tous les entry points MAPS (CLI, sweep
> drivers, notebooks). Format uniforme timestamp + name + level + msg.
>
> API :
> ```python
> configure_logging(level="INFO", log_file=None, *, force=True) -> Logger
> ```
>
> - `level` : str ("INFO", "DEBUG") ou int → résolu via
>   `logging.getLevelName`, raise si inconnu
> - `log_file` : optional Path. Si donné, ajoute un FileHandler (mode
>   append, parent dirs créés) en plus du StreamHandler stderr
> - `force=True` : passe-through `logging.basicConfig(force=True)` —
>   clear les handlers existants, utile pour re-config depuis un
>   notebook
>
> **Ne JAMAIS** silence ou désactive des progress bars / third-party
> loggers (lab convention CLAUDE.md : "I want to see everything").
>
> Format :
> ```
> 2026-05-25 11:48:32 maps.domains.blindsight.trainer [INFO] Starting training run, seed=42
> ```

## 2. Contraintes d'ingénierie

### Pourquoi stdlib `logging` et pas `loguru` / `structlog` ?

- **Stdlib** : 0 deps, intégration native avec PyTorch / NumPy /
  matplotlib (qui logguent aussi via stdlib)
- **loguru** : ergonomie supérieure mais redirige tout vers son sink,
  potentiel friction avec PyTorch DDP logging
- **structlog** : structured logging top-tier mais overkill pour un
  projet recherche

Convention lab : stdlib. Pas de débat.

### Format

`%(asctime)s %(name)s [%(levelname)s] %(message)s` est lisible et
parsable. `%(name)s` permet de filtrer par module
(`maps.domains.blindsight.trainer`). Pas de couleur (les logs vont
souvent vers fichier ou pipe).

### File handler

- Mode `"a"` (append) — re-runs cumulés dans le même fichier
- Parents créés via `parent.mkdir(parents=True, exist_ok=True)`
- Encoding `utf-8` explicit (sinon défault OS, peut casser sur Windows)

## 3. Deviations / dettes présentes

Pas de déviation. Module propre, simple, faisant exactement ce qui
est attendu.

CLAUDE.md hard rule "never silence console output" est respecté —
on n'ajoute pas de `setLevel(WARNING)` sur des loggers third-party.

## 4. Questions ouvertes

- **Q1 :** Doit-on standardiser un naming convention pour les loggers ?
  Currently `getLogger(__name__)` donne `maps.domains.blindsight.trainer`
  ce qui est OK. Mais on pourrait imposer un prefix uniforme pour le
  filtrage.

## 5. Pour le rewrite (Sprint 11+)

### À garder
- API simple, stdlib backend
- Format timestamp + name + level + msg
- File handler optional, append mode
- `force=True` default

### À améliorer
- **Rotating file handler** : pour les longs runs RL (SARL+CL,
  curriculum 4 envs), le log file peut grossir. Option
  `rotate_max_bytes` ?
- **Per-module level** : permettre `level={"torch": "WARNING",
  "maps": "INFO"}` pour réduire le bruit sans tout cacher
- **JSON sink** : option output `format="json"` pour parser logs en
  aval

### Tests à écrire
- `configure_logging("INFO")` configure le root logger
- `configure_logging("INFO", log_file=tmp_path / "x.log")` écrit dans
  le fichier
- Raise sur level invalide
- Idempotence (double appel ne casse pas)

### Connexion
- Appelé en tête de tous les `cli.py` (après `set_all_seeds`)
- Modules src/ utilisent `logger = logging.getLogger(__name__)`
