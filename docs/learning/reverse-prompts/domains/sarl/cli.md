# Reverse-prompt — `src/maps/domains/sarl/cli.py`

**Module path actuel (sur main) :** `scripts/run_sarl.py`
**Taille :** ~335 lignes.

---

## 1. Quelle aurait été la spec ?

> Écris un CLI typer qui lance UN training run SARL (game × setting ×
> seed). Plus complexe que Blindsight/AGL CLI car :
> 1. Setting 7 dispatch vers `ACBTrainer` (algorithme structurellement
>    différent — actor-critic)
> 2. Settings 1-6 dispatch vers `run_training(SarlTrainingConfig)`
> 3. Resume mécanisme (`--resume` auto-detect, `--resume-from path`
>    explicit)
> 4. Variant v1 / v2 dispatch (`--override training.model_variant=v2`)
>
> **Output layout** :
> `$SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<seed>/`
>
> **CLI args** :
> - `--game` : space_invaders, breakout, seaquest, asterix, freeway
> - `--setting N` (1-7) : paper setting
> - `--seed N`
> - `--num-frames N` : override config (default 500_000)
> - `--resume` : auto-detect `checkpoint.pt` in output_dir
> - `--resume-from path` : explicit checkpoint
> - `-o key=value` : Hydra-style override (repeatable)
> - `--output-dir` : override base
>
> **Dispatch logic** :
> ```python
> if setting == 7:
>     # ACB path
>     cfg_acb = load_config("training/sarl_acb")
>     acb_cfg = ACBConfig.for_game(game, seed=seed, ...)
>     trainer = ACBTrainer(env, network, acb_cfg)
>     trainer.train()
> else:
>     # Settings 1-6 — standard DQN
>     cfg = load_config("training/sarl", overrides=overrides)
>     sarl_cfg = _build_training_config(cfg, game, setting, seed, ...)
>     env = _build_env(game)
>     run_training(env, sarl_cfg)
> ```
>
> **Per-run save** (settings 1-6) :
> - `metrics.json` (training curves + validation summaries)
> - `policy_net.pt` (final state)
> - `target_net.pt`
> - `second_order_net.pt` (if meta)
> - `checkpoint.pt` (intermédiaire chaque N updates)
> - `summary.json` (z-score relevant metrics)
>
> Setting 7 : différent output schema (ACB-specific).

## 2. Contraintes d'ingénierie

### Dispatch sur `setting` early

Le dispatch settings 1-6 vs 7 doit être early (avant load config)
parce que les configs diffèrent : `training/sarl.yaml` vs
`training/sarl_acb.yaml`.

### `_build_env(game)` lazy import

```python
def _build_env(game):
    try:
        from minatar import Environment
    except ImportError as exc:
        raise RuntimeError("MinAtar not installed. uv sync --extra sarl")
    ...
    return Environment(game)
```

Lazy : si on lance le CLI sans `--extra sarl`, on a une erreur claire,
pas un cryptic import error au top du module.

### `_SUPPORTED_GAMES` whitelist

5 games MinAtar testés dans le paper. Validation early :
```python
if game not in _SUPPORTED_GAMES:
    raise typer.BadParameter(...)
```

### Resume auto-detect vs explicit

```python
if resume and resume_from is None:
    resume_from = output_dir / "checkpoint.pt"
    if not resume_from.exists():
        raise FileNotFoundError("--resume but no checkpoint.pt")
```

Auto = convenience pour SLURM jobs restartés. Explicit = test ou
debug.

### `_build_training_config` defensive

Utilise `getattr(cfg.training, "gamma", 0.999)` pour gérer les YAML
legacy qui n'ont pas tous les fields. Permet de garder le CLI
fonctionnel sur des configs anciennes.

## 3. Deviations / dettes présentes

Pas de déviation paper sur ce module (pure plumbing).

**Pas de DETTE active.**

## 4. Questions ouvertes

- **Q1 :** Le dispatch settings 1-6 vs 7 dans le CLI crée un peu de
  duplication (`run_training` vs `ACBTrainer.train`). Alternative :
  un common interface `TrainerProtocol` qui standardise. Mais ACB et
  MAPS-DQN sont vraiment des algorithmes différents — duplication
  acceptable.

- **Q2 :** Le `summary.json` schema n'est pas pydantic-versioned.
  Pour les aggregate scripts (Phase F), c'est fragile si on change un
  champ. Pydantic + versioning serait robuste.

## 5. Pour le rewrite (Sprint 11+)

### À garder
- Typer CLI standard
- Dispatch settings 1-6 vs 7
- Resume mechanism
- Variant v1/v2 dispatch
- `_SUPPORTED_GAMES` whitelist
- Lazy MinAtar import

### À améliorer
- **Co-localisation** : `scripts/run_sarl.py` → `domains/sarl/cli.py`,
  use `python -m maps.domains.sarl.cli`
- **Pydantic schema** pour summary.json
- **Validation cfg ↔ checkpoint** plus robust (versioning)
- **Logging structuré JSON** (au lieu de log.info text)

### Tests à écrire
- Smoke setting 1-6 : `python -m maps.domains.sarl.cli --game
  space_invaders --setting 1 --seed 42 --num-frames 1000` exit 0
- Setting 7 dispatch vers ACBTrainer
- Resume auto-detect
- Override v1 vs v2 dispatch
- Resume avec cfg mismatch raise

### Connexion
- Utilise : `domains.sarl.{ACBConfig, ACBTrainer, SarlTrainingConfig,
  run_training, setting_to_config}`
- Utilise : `utils.{configure_logging, get_paths, load_config,
  set_all_seeds}`
- Configure via : `config/training/sarl.yaml`, `config/training/sarl_acb.yaml`
