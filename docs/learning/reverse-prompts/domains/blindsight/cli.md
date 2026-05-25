# Reverse-prompt — `src/maps/domains/blindsight/cli.py`

**Module path actuel (sur main) :** `scripts/run_blindsight.py`
(co-localisé en `domains/blindsight/cli.py` au Sprint 11+).
**Taille :** ~210 lignes.

---

## 1. Quelle aurait été la spec ?

> Écris un CLI typer pour lancer un (ou plusieurs) run(s) Blindsight.
> Le module charge la config composée
> (`config/training/blindsight.yaml` ← `config/maps.yaml`), seed
> toutes les RNG, build les networks, run pre-training, save loss
> curves + final state.
>
> **Output layout** :
> `$SCRATCH/maps/outputs/blindsight/<setting>/seed-<seed>/` sur
> HPC, fallback `./outputs/blindsight/<setting>/seed-<seed>/` en dev.
>
> **Args CLI typer** :
> - `--setting setting-X-name` : factorial setting id (default
>   "setting-6-full-maps")
> - `--factorial-config path` : YAML défini les settings + seeds
>   (default `experiments/factorial_6cell` paper Table 5,
>   alternative `experiments/factorial_2x2` archive parity)
> - `--all-settings` : loop sur tous les settings × seeds
> - `--seed N` : override seed (single-setting mode)
> - `--seeds 42,43,...` : liste explicite (--all-settings mode)
> - `-o key=value` (repeatable) : Hydra-style override
> - `--output-dir path` : override base out
> - `--log-level INFO` : Python logging level
>
> **Per-run save** :
> - `losses_1.npy` (CAE loss curve)
> - `losses_2.npy` (BCE wager loss curve)
> - `first_order.pt` (final state dict)
> - `second_order.pt` (final state dict)
> - `summary.json` (setting, seed, n_epochs, final losses, eval metrics,
>   timings)
>
> **Resolve seed pool** (CLI > YAML explicit > YAML n_seeds count) :
> ```python
> if seeds_cli is not None:
>     pool = parse(seeds_cli)
> elif yaml.seeds is not None:
>     pool = yaml.seeds
> else:
>     pool = list(range(yaml.n_seeds))
> ```
>
> **SLURM array** : production utilise `scripts/slurm/blindsight_array.sh`
> qui lance N processus, chacun avec un seed différent. Single-process
> loop est pour dev.

## 2. Contraintes d'ingénierie

### Typer vs argparse

Typer = wrapper FastAPI-style autour argparse. Avantages :
- Type hints natifs → auto-validation + auto-help
- Subcommands faciles (potentiel futur)
- Generation auto de help text rich
- Cohérence visuelle avec FastAPI / pydantic

Inconvénient : 1 dep en plus (déjà acceptée dans pyproject).

### `sys.path.insert(0, ...)` au top

Hack legacy pour permettre `python scripts/run_blindsight.py` sans
`pip install -e .`. Au Sprint 11+ avec le CLI co-localisé dans
`src/maps/domains/blindsight/cli.py`, on peut faire
`python -m maps.domains.blindsight.cli` proprement, plus besoin de
ce hack.

### Output dir resolution

Pattern : CLI override > `$SCRATCH/maps/outputs/blindsight` (HPC) >
`./outputs/blindsight` (dev). Via `paths.scratch_root`. Cohérent
cross-environnement.

### Logging config

`configure_logging(level=log_level)` en tête. Log to stderr by default.
Pas de file handler par run — tu peux ajouter avec `--log-level` +
shell redirection.

### Atomicité

Chaque run :
1. `out_dir.mkdir(parents=True, exist_ok=True)` (idempotent)
2. Run training
3. Save tous les artifacts (numpy + torch.save + summary.json)
4. Log info

Pas de transaction explicite ; si interrupt durant save, certains
fichiers peuvent être présents et d'autres pas. Acceptable pour
research code.

## 3. Deviations / dettes présentes

Pas de déviation paper sur ce module (pure plumbing).

**Pas de DETTE active** sur le code. Au Sprint 11+ :
- Sortir du `sys.path.insert` hack (utiliser `python -m`)
- Co-localiser dans `domains/blindsight/cli.py`

## 4. Questions ouvertes

- **Q1 :** Pourquoi pas un global CLI `maps blindsight ...` au lieu
  d'un CLI par domaine ? L'archi actuelle (un script par run_X.py)
  duplique le boilerplate (config load, paths, seeding, output dir).
  Option pour Sprint 11+ : un `maps/cli/main.py` + sous-commandes.

- **Q2 :** Comment standardiser le summary.json schema across domains ?
  Actuellement chaque domaine a son propre format. Convention commune
  faciliterait les aggregate scripts.

## 5. Pour le rewrite (Sprint 11+)

### À garder
- Typer pour les args, dataclass-style
- Output layout `$SCRATCH/maps/outputs/<domain>/<setting>/seed-<seed>/`
- Save artefacts standardisés (losses, models, summary)
- Hydra-style override `-o key=value`
- Mode `--all-settings` pour loop dev local

### À améliorer
- **Plus de `sys.path.insert`** : utiliser `python -m maps.domains.X.cli`
- **Co-localisation** : `domains/blindsight/cli.py` au lieu de
  `scripts/`
- **Optional : global CLI `maps/cli/main.py`** qui dispatch :
  `maps blindsight --setting X --seed Y`, `maps agl ...`, etc.
- **Schema commun pour summary.json** : pydantic model
  `RunSummary` partagé entre domaines
- **Error handling** : try/except autour de `_run_one` avec retry sur
  certains errors (CUDA OOM ?), checkpoint partial state

### Tests à écrire
- CLI integration : `python -m maps.domains.blindsight.cli --setting
  setting-1-baseline --seed 42 -o train.n_epochs=2` exit 0
- Output structure : tous les fichiers attendus présents après run
- `--all-settings` itère sur tous les settings
- `--seeds 42,43,44` parsing correct
- `-o train.n_epochs=5` override respecté

### Connexion
- Utilise `domains.blindsight.{BlindsightSetting, BlindsightTrainer}`
- Utilise `utils.{configure_logging, get_paths, load_config,
  set_all_seeds}`
- Lancé par SLURM array scripts (post-Sprint 15)
