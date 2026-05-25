# Reverse-prompt — `src/maps/domains/agl/cli.py`

**Module path actuel (sur main) :** `scripts/run_agl.py`
**Taille :** ~290 lignes (plus gros que Blindsight CLI car 4-phase output).

---

## 1. Quelle aurait été la spec ?

> Écris un CLI typer pour lancer le full 4-phase protocole AGL :
>
> 1. **Pretrain** (random grammar) — `trainer.pre_train()`
> 2. **Replicate** post-pretrain dans 20-cell `AGLNetworkPool`
> 3. **Training** : High tier 12 epochs, Low tier 3 epochs Grammar-A
> 4. **Test** : `trainer.evaluate_pool(pool)` → metrics par tier
>
> **Output layout** :
> `$SCRATCH/maps/outputs/agl/<setting>/seed-<seed>/`
>
> **Artefacts saved** :
> - `pretrain_losses_1.npy` (1st-order CAE loss curve)
> - `pretrain_losses_2.npy` (2nd-order BCE loss curve)
> - `first_order_reset.pt` (post-reset 1st-order state)
> - `second_order_postpre.pt` (post-pretrain 2nd-order state)
> - `training_high_losses_1.npy` (10 cells × 12 epochs)
> - `training_high_losses_2.npy`
> - `training_high_precision.npy`
> - `training_low_losses_1.npy` (10 cells × 3 epochs)
> - `training_low_losses_2.npy`
> - `training_low_precision.npy`
> - `summary.json` (nested per-phase metrics)
>
> **CLI args** : identique à Blindsight CLI (typer args, factorial
> config 6-cell vs 2x2, seed pool, --all-settings, --override -o).
>
> **Summary JSON schema** :
> ```json
> {
>   "setting": "setting-4-maps",
>   "seed": 42,
>   "num_networks": 20,
>   "pretrain": {n_epochs, loss_1_final, loss_2_final, loss_1_min,
>                loss_2_min, elapsed_seconds},
>   "training_high": {n_epochs, n_cells, precision_final_mean,
>                     loss_1_final_mean, elapsed_seconds},
>   "training_low": {idem},
>   "evaluation": {
>     "high": {precision_1st, precision_1st_std, wager_accuracy, ...,
>              precision_2nd, recall_2nd, f1_2nd, ...},
>     "low": {idem},
>     "overall": {idem},
>     "elapsed_seconds": ...
>   },
>   "meta_frozen_in_training": true
> }
> ```

## 2. Contraintes d'ingénierie

### Sequential phases vs streaming

Les 4 phases s'exécutent séquentiellement dans un seul process. Pas de
checkpoint intermédiaire (sauf les .npy/.pt saved). Si crash pendant
training_low, on perd training_high partial work.

Améliration potentielle Sprint 11+ : save intermédiaire après chaque
phase, resume from checkpoint.

### Logging par phase

`log.info` après chaque phase avec metrics clés (elapsed + accuracy
proxies). Permet de suivre la progression en tail des logs.

### Phase 4 = evaluate_pool, pas just evaluate

Important pour reproduire paper : Phase 4 utilise `evaluate_pool` (sur
les 20 cells avec aggregation per tier), pas `evaluate` (single cell).
La différence est dans le `cli.py` : on appelle
`trainer.evaluate_pool(pool)`.

### Co-localisation au Sprint 11+

Au Sprint 11+, déplacer `scripts/run_agl.py` →
`src/maps/domains/agl/cli.py`. Use `python -m maps.domains.agl.cli`.

## 3. Deviations / dettes présentes

Pas de déviation paper sur ce module (pure plumbing CLI).

**Pas de DETTE active** sur le code.

## 4. Questions ouvertes

- **Q1 :** Pas de resume from checkpoint. Pour un run de 4 phases
  avec total ~2-5 min sur Blindsight/AGL, c'est acceptable. Mais pour
  des sweeps de 500 seeds, ça commence à compter. Faut-il un
  mécanisme de skip-if-summary-exists ?
- **Q2 :** Le summary JSON est nested. Pour des aggregations
  cross-seeds, on parse souvent ce JSON. Pourrait-on émettre aussi un
  CSV flat pour faciliter aggregation pandas ?

## 5. Pour le rewrite (Sprint 11+)

### À garder
- Sequential 4-phase pipeline
- Save tous les artefacts (.npy + .pt + summary.json)
- Nested summary schema clair (pretrain / training_high / training_low
  / evaluation)
- typer CLI standard
- Factorial 6-cell + legacy 2x2 support

### À améliorer
- **Resume from checkpoint** : skip phase si artefact présent (avec
  flag `--force` pour override)
- **Co-localisation** : `domains/agl/cli.py`, `python -m maps...`
- **Pydantic schema pour summary** : `AGLRunSummary` BaseModel
  versioned + emit JSON + CSV
- **Phase-level timing logs** : structurer en JSON records
- **Error handling** : retry on transient errors (CUDA OOM), partial
  recovery

### Tests à écrire
- CLI integration : `python -m maps.domains.agl.cli --setting
  setting-1-baseline --seed 42 -o train.n_epochs_pretrain=2
  -o train.n_epochs_training_high=2
  -o train.n_epochs_training_low=1` exit 0
- All artefacts présents après run
- `--all-settings` itère tous les settings × seeds
- Override sur n_epochs respecté

### Connexion
- Utilise : `domains.agl.{AGLSetting, AGLTrainer, AGLNetworkPool}`
- Utilise : `utils.{configure_logging, get_paths, load_config,
  set_all_seeds}`
- Output schema partagé conceptuellement avec Blindsight (pas de
  pydantic shared yet)
