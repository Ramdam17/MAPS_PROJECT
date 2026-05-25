# Reverse-prompt — `src/maps/domains/sarl_cl/cli.py`

**Module path actuel (sur main) :** `scripts/run_sarl_cl.py`
**Taille :** ~335 lignes.

---

## 1. Quelle aurait été la spec ?

> Écris un CLI typer pour SARL+CL. Mirror `scripts/run_sarl.py` mais
> avec **3 CLI flags supplémentaires pour CL** :
>
> - `--curriculum` : enable 3-term CL loss (task + weight_reg + feature)
> - `--adaptive` : use `AdaptiveQNetwork` (variable in_channels)
> - `--teacher-load-path` : path to a previous-task checkpoint
>
> **Curriculum chaining via CLI** :
>
> ```bash
> # Stage 1 : Breakout from scratch
> python -m maps.domains.sarl_cl.cli --game breakout --setting 6
>
> # Stage 2 : Space Invaders with Breakout teacher
> python -m maps.domains.sarl_cl.cli --game space_invaders --setting 6 \
>     --curriculum --adaptive \
>     --teacher-load-path outputs/sarl_cl/breakout/setting-6/seed-42/checkpoint.pt
>
> # Stage 3 : Seaquest with SI teacher
> python -m maps.domains.sarl_cl.cli --game seaquest --setting 6 \
>     --curriculum --adaptive \
>     --teacher-load-path outputs/sarl_cl/space_invaders/.../checkpoint.pt
>
> # Stage 4 : Freeway with Seaquest teacher
> # ... etc.
> ```
>
> **Resume mécanisme** : same as SARL — `--resume` auto-detect,
> `--resume-from path` explicit.
>
> **Output layout** :
> `$SCRATCH/maps/outputs/sarl_cl/<game>/setting-<N>/seed-<seed>/`
>
> **CLI flags win over YAML** pour les CL toggles — keeps curriculum
> decisions visible at command line, not buried in config.

## 2. Contraintes d'ingénierie

### CLI flags > YAML for CL

Les 3 CL flags (`--curriculum`, `--adaptive`, `--teacher-load-path`)
**override toujours** la YAML. Important pour SLURM array scripts qui
veulent contrôler le curriculum explicitement.

### `_build_training_config` builds `SarlCLTrainingConfig`

Mirror `_build_training_config` from `run_sarl.py` mais retourne
`SarlCLTrainingConfig` avec les 3 CL fields populated.

### `_SUPPORTED_GAMES` whitelist same as SARL

Re-use la même whitelist : 5 MinAtar games. Pas de différence pour
CL.

### Resume avec teacher

Si on resume un run CL stage 2, on reload :
1. Le `checkpoint.pt` du run (policy_net, target_net, etc.)
2. Le `teacher_load_path` (teacher_first_net, teacher_second_net)

Les 2 chemins sont préservés dans le checkpoint resume.

## 3. Deviations / dettes présentes

Pas de déviation paper sur ce module.

**Pas de DETTE active.**

## 4. Questions ouvertes

- **Q1 :** Curriculum chaining via SLURM dependent jobs : `sbatch
  --dependency=afterok:JOBID1 stage2.sh`. C'est la voie HPC. Doc
  needed.

- **Q2 :** `--teacher-load-path` est un Path local. Sur HPC avec
  `$SCRATCH` partagé, c'est OK. Sur dev local cross-machine, fragile.

## 5. Pour le rewrite (Sprint 11+)

### À garder
- 3 CL flags + resume mechanism
- Output schema cohérent avec SARL standard
- Mirror SARL CLI structure

### À améliorer
- **Co-localisation** : `domains/sarl_cl/cli.py`, `python -m`
- **Curriculum helper script** : `scripts/run_curriculum.sh` qui
  chain les 4 stages
- **Schema versioning** : pydantic pour summary.json + checkpoint
  metadata

### Tests à écrire
- Smoke single-task (no teacher)
- Smoke multi-task (with teacher)
- Resume from checkpoint
- AdaptiveQNetwork dispatch

### Connexion
- Utilise : `domains.sarl_cl.{SarlCLTrainingConfig, run_training_cl}`
- Utilise : `utils.{configure_logging, get_paths, load_config,
  set_all_seeds}`
- Configure via `config/domains/sarl_cl/training.yaml`
- Curriculum via SLURM dependent jobs (Phase F)
