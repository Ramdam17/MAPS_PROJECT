# Sprint 03 — Config-first migration

**Status:** ⚪ planned
**Branch:** `refactor/config-first`
**Owner:** Rémy Ramadour
**Est. effort:** 2-3 days
**Depends on:** Sprint 02

---

## Context

The @config-auditor found ~100 hardcoded hyperparameters and 5 shell scripts with absolute paths pointing to 3 different authors' home directories. Moving all of this to YAML removes the "lab's laptop" assumption and makes experiments declarative.

We use **OmegaConf + Hydra-style composition** (already the convention in METTA). Not Hydra itself — we use OmegaConf directly to keep dependencies minimal and composition explicit.

## Objectives

1. Zero hardcoded path, seed, learning rate, hidden dim, or env name in `src/maps/*`
2. Experiments declared as YAML overlays, not shell argument matrices
3. Single `typer` CLI per domain that takes `--config path/to/config.yaml` + dotted overrides

## Tasks

### 3.1 — Config schemas
- [ ] `config/maps.yaml` — canonical constants (from `CLAUDE.md` table)
- [ ] `config/paths.yaml` — host-dependent: `repo_root`, `data_dir`, `output_dir`, `checkpoint_dir`, `log_dir`
    - [ ] default for Mac: `outputs/`, `logs/`, `models/`, `data/`
    - [ ] override for Narval: `$SCRATCH/maps/...`
    - [ ] override for Rorqual: same pattern
- [ ] `config/training/blindsight.yaml`
- [ ] `config/training/agl.yaml`
- [ ] `config/training/sarl_dqn.yaml`
- [ ] `config/training/sarl_cl.yaml`
- [ ] `config/training/marl_mappo.yaml`
- [ ] `config/env/minatar.yaml` — per-substrate block (5 envs)
- [ ] `config/env/meltingpot.yaml` — per-substrate block (20+ envs, only 4 used by paper)
- [ ] `config/experiments/` — one YAML per "setting" from paper (settings 1-6), each overlaying the base training config

### 3.2 — Python loader
- [ ] `src/maps/config.py`:
    - [ ] `load_config(path, overrides=None) -> OmegaConf.DictConfig`
    - [ ] Supports `${oc.env:VAR,default}` interpolation
    - [ ] Supports `_BASE_: path/to/parent.yaml` composition
    - [ ] Resolves paths relative to repo root, not cwd
- [ ] `src/maps/paths.py`:
    - [ ] `get_paths() -> Paths` (dataclass or DictConfig)
    - [ ] Reads `config/paths.yaml` or `$MAPS_PATHS_CONFIG` env var
    - [ ] Fails loudly if `outputs/` etc. are not writable

### 3.3 — CLI entry points
- [ ] `scripts/train_blindsight.py`:
    ```python
    import typer
    from maps.config import load_config
    app = typer.Typer()
    @app.command()
    def train(config: Path, seed: int = 42, setting: int = 4, overrides: list[str] = None):
        cfg = load_config(config, overrides)
        ...
    ```
- [ ] `scripts/train_agl.py` — same shape
- [ ] `scripts/train_sarl.py` — loads `config/training/sarl_dqn.yaml` + `config/env/minatar.yaml#<substrate>`
- [ ] `scripts/train_sarl_cl.py`
- [ ] `scripts/train_marl.py` — Linux-only, guarded
- [ ] Register in `pyproject.toml`:
    ```toml
    [project.scripts]
    maps-train-blindsight = "maps.scripts.train_blindsight:app"
    # etc.
    ```

### 3.4 — Shell scripts
- [ ] Rewrite `scripts/slurm/*.sh` (keep 1 template) that reads `$MAPS_CONFIG` + `$MAPS_PATHS_CONFIG` and dispatches to the right `python -m` entry
- [ ] Delete `SARL_Training_*.sh`, `meltingpot.sh`, `SARL_Plot_Results.sh` at end of sprint — their functionality is subsumed
- [ ] Keep `MARL/MAPPO-ATTENTIOAN/{eval,train}_meltingpot.sh` for now (Sprint 07 problem)

### 3.5 — Dotted-override CLI support
- [ ] `uv run maps-train-blindsight --config config/training/blindsight.yaml \
       seed=7 maps.cascade_iterations=100 training.epochs=3`
  (standard OmegaConf `key=value` syntax)

## Definition of Done

- `grep -rnE "/(home|Users)/" src/ scripts/ config/` returns nothing
- `grep -rnE "0\.00025|cascade.*50|iterations.*50" src/maps/` returns nothing (all values in YAML)
- `uv run maps-train-blindsight --config config/training/blindsight.yaml --dry-run` prints the fully resolved config tree
- All hyperparameters listed in `docs/TODO.md` config-auditor report are externalised
- `uv run pre-commit run --all-files` passes (the `ban-abs-home-paths` hook would block any regression)

## Risks

- Paper uses 6 "settings" as an axis — naive mapping ⇒ 6 YAML files per domain = 30 configs. Instead, parametrize as `experiments.settings: list[{id, second_order_on, cascade_on, extra}]` in a single file per domain.
- MeltingPot substrate parameters are already in giant shell `case` blocks — direct port to YAML will produce one file per substrate. Fine.
