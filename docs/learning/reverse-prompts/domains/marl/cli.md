# Reverse-prompt — `src/maps/domains/marl/cli.py`

**Module path actuel :** `scripts/run_marl.py`
**Taille :** ~310 lignes.
**Linux-only** (deps `.venv-marl` Python 3.11).

---

## 1. Quelle aurait été la spec ?

> Écris CLI typer pour MARL. Charge `config/domains/marl/training.yaml` +
> `config/domains/marl/env/<substrate>.yaml` + `config/experiments/factorial_marl.yaml`,
> build MeltingPot env + separated-MAPPO runner, trains one cell
> (substrate × setting × seed).
>
> **Output layout** :
> `$SCRATCH/maps/outputs/marl/<substrate>/setting-<id>/seed-<seed>/`
>
> **CLI args** :
> - `--substrate` : commons_harvest_closed, commons_harvest_partnership,
>   chemistry, territory_inside_out (4 substrates)
> - `--setting` : baseline, cascade_1st_no_meta, meta_no_cascade, maps,
>   meta_cascade_2nd, meta_cascade_both (6 settings)
> - `--seed`
> - `--num-env-steps` : override (default 300k paper §4 text)
> - `-o key=value` Hydra-style
> - `--output-dir` override
>
> **Invocation** :
> ```bash
> # Requires .venv-marl
> .venv-marl/bin/python scripts/run_marl.py \
>     --substrate commons_harvest_closed --setting maps --seed 42
>
> # Smoke test
> .venv-marl/bin/python scripts/run_marl.py \
>     --substrate territory_inside_out --setting baseline \
>     --seed 42 --num-env-steps 10000
> ```
>
> **SLURM array production** :
> 6 settings × 4 substrates × 3 seeds = **72 cells**. Submitted via
> `scripts/slurm/marl_array.sh` avec `--gres=gpu:h100:1` (ou h200:1)
> et ~24h wall.

## 2. Contraintes d'ingénierie

### `.venv-marl` constraint

dmlab2d / meltingpot ne build pas sur Python 3.12 ni macOS. Donc
nécessite `.venv-marl` (Python 3.11 + native libs). Documented dans
`docs/install_marl_drac.md`.

### `_SUPPORTED_SUBSTRATES` whitelist

4 substrates qui matchent paper Table 7. Pas plus.

### `_SUPPORTED_SETTINGS` whitelist

6 settings de la factorial. Validation early via typer.BadParameter.

### Multi-file config compose

3 configs YAML :
- `config/domains/marl/training.yaml` : hyperparams (Table 12)
- `config/domains/marl/env/<substrate>.yaml` : env-specific (num_agents,
  episode_length, max_cycles)
- `config/experiments/factorial_marl.yaml` : 6 settings

Composed via OmegaConf merge.

## 3. Deviations / dettes présentes

Pas de déviation paper.

**Pas de DETTE active.**

## 4. Questions ouvertes

- **Q1 :** `.venv-marl` constraint = dev local impossible sur Mac.
  Frustrant. Mock env Mac-compatible serait gold.

- **Q2 :** Workflow SLURM array : 72 cells × 24h = 72 × 24 = 1728
  GPU-hours. Compute Canada budget significant.

## 5. Pour le rewrite (Sprint 11+)

### À garder
- Multi-file config compose
- Whitelist substrates + settings
- typer CLI standard
- $SCRATCH-aware output

### À améliorer
- **Co-localisation** : `domains/marl/cli.py`, `python -m`
- **Mock env support** : `--substrate mock` for Mac dev
- **Schema versioning** pour summary.json

### Tests à écrire
- Smoke avec mock env
- Substrate whitelist
- Setting whitelist
- Resume from checkpoint

### Connexion
- Utilise : `domains.marl.{MarlSetting, MeltingpotRunner, RunnerConfig}`
- Utilise : `domains.marl.env.build_env_from_config`
- Utilise : `utils.{configure_logging, get_paths, load_config,
  set_all_seeds}`
- Lancé par : SLURM array Phase F MARL
