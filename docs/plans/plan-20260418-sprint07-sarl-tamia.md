# Plan: Sprint 07 — SARL reproduction sur Tamia (consolidation → profil → optims → run)

**Date:** 2026-04-18
**Sprint:** Sprint-07
**Cluster cible:** Tamia (DRAC) — H100, allocation `aip-gdumas85`
**Fallback:** Narval — A100, allocation `def-gdumas85_{cpu,gpu}`
**Estimated complexity:** L (1-2 j ingénierie + 2-6 h compute en régime optimiste)
**Branche de travail:** `repro/sarl` (depuis `main`)
**Authoring:** Rémy Ramadour + Claude
**Status:** validé (décisions verrouillées 2026-04-18) — prêt à exécuter Phase 0

---

## Problem Statement

Sprint 04b a livré une parité architecturale SARL à atol=1e-6 (CPU). Sprint 07 doit produire la parité **scientifique** : reproduire les z-scores SARL du papier (Vargas et al., TMLR) sur au moins 2 des 5 environnements MinAtar, à ±2σ, via **300 cellules** (5 jeux × 6 settings × **10 seeds** × 5M frames) sur Tamia. (Décision 2026-04-18 : N=10 seeds, conforme au papier ; supersede le sprint-07 spec qui disait N=5.)

Blockers identifiés en discussion (2026-04-18) :

1. **Consolidation repo** : dossiers legacy (`SARL/`, `SARL_CL/`, `MARL/MAPPO-ATTENTION/`, `requirements-legacy.txt`) toujours trackés contrairement à ce que dit le rapport Sprint 04b. Constantes dupliquées (`training_loop.py:74-83` vs yaml). Scaffolding SLURM minimal, non calibré, pas de routage `$SCRATCH`.
2. **Inconnus perf** : la branche "perf notes" qu'on espérait inspecter n'existe pas sur origin (les 3 branches `refactor/*` sont pre-Sprint-04b). On part donc sur **profiling-first** — Phase 2 produit la vérité terrain.
3. **Cluster nouveau** : Tamia jamais utilisé dans ce repo. Le code n'a jamais tourné sur H100. Parité testée uniquement CPU.
4. **Budget** : Tamia/aip-* devrait drastiquement réduire le wall-clock (priorité haute), mais à confirmer par mesure.

**Contraintes dures :**
- Parité Tier 1/2/3 SARL à atol=1e-6 sur CPU maintenue après toute optimisation.
- Constantes MAPS (`config/maps.yaml`) lockées.
- Pas de re-seeding silencieux, pas de pivot méthodo, pas de feature-flag de compromis.
- Outputs SARL compatibles avec `scripts/aggregate_sarl.py` (schéma existant).

## Scientific Rationale

**Ordre d'exécution :**

1. D'abord un run témoin court (smoke) qui passe sur Tamia, sur une cellule, du bout-en-bout.
2. Ensuite un benchmark (cellule 500 k frames, H100) → FPS réel, calibration `--time`/`--gres`.
3. Profil `cProfile` + `torch.profiler` → liste chiffrée de bottlenecks.
4. Optimisations **parity-safe uniquement**, re-testées après chaque commit.
5. Lancement de la grille 300 cellules (5 games × 6 settings × 10 seeds).
6. Agrégation + rapport.

Rationale : sans rapport reproductible, toute "optim" peut expliquer un écart de z-score a posteriori. *"Premature optimization is the root of all evil"* (Knuth) — on profile avant d'optimiser.

## Implementation Approach

### Data flow (vue globale)

```
Phase 0  — setup Tamia + validation env              [0.5 j]
  ↓
Phase 1  — consolidate (legacy + SLURM + paths HPC)  [0.5–1 j]
  ↓
Phase 2  — smoke → bench → profile                   [0.5 j compute + 1–4 h queue]
  ↓
Phase 3  — optims parity-safe (conditionnel)         [0.5 j]
  ↓
Phase 4  — array 300 cells                           [4 h–4 j wall-clock]
  ↓
Phase 5  — aggregation + rapport                     [0.5 j]
```

---

### Phase 0 — Setup Tamia & validation env (0.5 j, **bloquant**)

**But :** avoir un checkout propre sur Tamia, env Python synché, SSH GitHub actif, sanity check des tests unitaires et de parité.

**Tâches :**
- [ ] P0.1 — Sur Tamia : `ssh-keygen -t ed25519 -C "rram17@tamia"`, ajouter `~/.ssh/id_ed25519.pub` à GitHub. `ssh -T git@github.com` → doit dire "Hi Ramdam17!".
- [ ] P0.2 — `cd $HOME && git clone git@github.com:Ramdam17/MAPS_PROJECT.git`. Créer branche `repro/sarl` depuis `main`.
- [ ] P0.3 — Vérifier l'allocation : `sshare -U -u rram17` sur Tamia → doit montrer `aip-gdumas85_{cpu,gpu}`. Noter les RawShares / usage.
- [ ] P0.4 — Recenser les partitions GPU Tamia : `sinfo -o "%P %G %l"` + `scontrol show partition | grep -E "PartitionName|MaxTime"`. Spécifiquement : présence de MIG H100, tranche max (`MaxTime`), quotas aip-*. Consigner dans `docs/install_tamia.md` (à créer en Phase 1).
- [ ] P0.5 — `module load StdEnv/2023 python/3.12 cuda/12.6` puis `curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"`.
- [ ] P0.6 — `cd MAPS_PROJECT && uv sync --extra sarl --extra dev`. Surveiller l'installation PyTorch : wheel CUDA doit matcher cuda/12.6.
- [ ] P0.7 — Smoke env : `uv run python -c "import torch, minatar; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"` → attendu `True, ≥1`.
- [ ] P0.8 — Smoke tests : `uv run pytest -q tests/unit tests/parity/sarl` → doit passer intégralement. Si ça casse, STOP, flag, fallback Narval.
- [ ] P0.9 — Smoke training bout-en-bout (interactif, pas sbatch, 1 cellule, 10k frames, CPU) : `uv run python scripts/run_sarl.py --game breakout --setting 1 --seed 42 --num-frames 10000 -o device=cpu` → doit produire `metrics.json` en <5 min.

**Deliverables :**
- Clé SSH Tamia active, GitHub.
- `~/MAPS_PROJECT` cloné, branche `repro/sarl`, `.venv/` OK.
- `docs/install_tamia.md` (squelette avec modules + allocation + partitions).

**DoD :**
- `git status` clean sur `repro/sarl`.
- `pytest tests/unit tests/parity/sarl` vert.
- Smoke CPU bout-en-bout : `outputs/sarl/breakout/setting-1/seed-42/metrics.json` non vide.
- `uv run python -c "import torch; assert torch.cuda.is_available()"` OK.

**Risques :**
- **R0-1** Clé SSH Tamia→GitHub : host key acceptance + firewall. *Fallback* : HTTPS + PAT GitHub.
- **R0-2** Wheel PyTorch CUDA 12.6 pas exactement aligné : uv peut tenter de compiler. *Mitigation* : pinner via `[tool.uv.sources]` si nécessaire.
- **R0-3** Tamia `aip-*` pas encore activé côté SLURM (Guillaume a ajouté le user, propagation peut prendre <24h). *Mitigation* : si P0.3 montre def-* seulement, attendre et/ou tester avec un sbatch `--test-only`.

**Effort :** 3–5 h active + debug potentiel.

---

### Phase 1 — Consolidate (0.5–1 j)

**But :** éliminer les dettes qui vont mordre en Phase 4. Aucune modif scientifique.

**1.A — Legacy cleanup**
- [ ] P1.1 — Confirmer avec Rémy : `SARL/`, `SARL_CL/`, `MARL/MAPPO-ATTENTION/`, `requirements-legacy.txt` → suppression totale oui/non ?
- [ ] P1.2 — Si go : `git rm -r SARL/ SARL_CL/ MARL/MAPPO-ATTENTION/ requirements-legacy.txt` + corriger `docs/TODO.md` (supprimer entrées TD pointant sur ces fichiers : TD-003, TD-004, TD-007, TD-008, TD-009, TD-018, TD-022, TD-040, etc.) + corriger `docs/reports/sprint-04b-report.md` (la mention "10,509 LOC removed" est partielle — préciser ce qui reste).
- [ ] P1.3 — Grep de clôture : `rg -l "juan-david-vargas|ubunto|/home/ubunto"` → 0 résultat.

**1.B — SLURM scaffolding Tamia-ready**
- [ ] P1.4 — Étendre `config/paths.yaml` : nouveau champ `scratch_root` (lu depuis `$SCRATCH` sur HPC, fallback `./outputs` en dev). Modifier `src/maps/utils/paths.py` pour consommer cette variable + test unitaire.
- [ ] P1.5 — Créer `scripts/slurm/common.sh` : source commune (modules, venv activate, `PYTHONUNBUFFERED=1`, `mkdir -p logs/slurm`, trap pour log signaux). Sourcé par tous les sbatch.
- [ ] P1.6 — Créer `scripts/slurm/smoke_sarl.sh` : 1 cellule × 50 k frames, paramétrable CPU/GPU, `--time=01:00:00`, `--account=aip-gdumas85_gpu` par défaut.
- [ ] P1.7 — Créer `scripts/slurm/bench_sarl.sh` : 1 cellule × 500 k frames, 3 modes (cpu_4c | gpu_full | gpu_mig si dispo sur Tamia). Écrit `outputs/bench/bench-<mode>-<rev>.json` avec {frames_per_s, updates_per_s, wall_s, peak_vram_mb, peak_rss_mb, seed, rev=git_sha}.
- [ ] P1.8 — Réécrire `scripts/slurm/sarl_array.sh` pour Tamia : `--account=aip-gdumas85_gpu`, `module load StdEnv/2023 python/3.12 cuda/12.6`, `--time`/`--gres`/`--array=0-299%N` lus depuis `bench.json` (ou défauts conservateurs). Ajout `--requeue` pour résilience pré-emption.
- [ ] P1.9 — Créer `scripts/slurm/monitor.sh` : one-shot `squeue --me` formatté + récap cellules manquantes (`outputs/sarl/<game>/setting-<N>/seed-<SEED>/metrics.json` ou absent).
- [ ] P1.10 — Compléter `docs/install_tamia.md` : section "aip-gdumas85 specifics" (quotas, partitions GPU, limite temps max, recommandations H100).

**1.C — Nettoyage interne (propreté, non bloquant)**
- [ ] P1.11 — Retirer les constantes dupliquées de `src/maps/experiments/sarl/training_loop.py:74-83` — le yaml est source de vérité, les defaults dataclass peuvent pointer directement vers OmegaConf. À prioriser uniquement si le refactor tient en < 1 h ; sinon décaler à un sprint "hygiène".
- [ ] P1.12 — Harmoniser `run_sarl.py` : choisir `-o k=v` comme chemin unique de surcharge, les `--flag` dédiés (`--num-frames`, `--output-dir`) deviennent des raccourcis documentés dans la docstring.

**Deliverables :**
- `scripts/slurm/{common,smoke_sarl,bench_sarl,sarl_array,aggregate,monitor}.sh`.
- `config/paths.yaml` étendu + `paths.py` + test unitaire nouveau.
- `docs/install_tamia.md` complet.
- Repo legacy-free (si P1.1 go).

**DoD :**
- `sbatch --test-only scripts/slurm/{smoke,bench,sarl_array}.sh` parsent sans erreur avec `--account=aip-gdumas85_gpu`.
- `uv run pytest -q tests/unit/utils/test_paths.py` vert.
- `rg "SARL/|SARL_CL/|MAPPO-ATTENTION" --files` = 0 si P1.1 go.
- `docs/install_tamia.md` couvre : clone, modules, uv, sbatch, compte, partitions, MIG si applicable.

**Risques :**
- **R1-1** Modifier `paths.py` casse des tests → on écrit le test NOUVEAU d'abord (TDD léger).
- **R1-2** Partitions Tamia différentes de Narval (pas de garantie que `gpubase_bynode_*` existe sur Tamia) → adapter d'après `sinfo` réel de P0.4.
- **R1-3** Suppression legacy (P1.2) = diff massif, revue à faire soigneusement (1 commit par dossier pour faciliter le review).

**Effort :** 5–8 h selon décision 1.A et 1.C.

---

### Phase 2 — Smoke → Benchmark → Profile (0.5 j + queue, **bloquant pour Phase 3/4**)

**But :** mesurer avant d'optimiser. Aucune optim sans preuve chiffrée.

**2.A — Smoke SLURM (valide le pipeline sous SLURM)**
- [ ] P2.1 — `sbatch --account=aip-gdumas85_cpu scripts/slurm/smoke_sarl.sh breakout 6 42 50000 cpu` → attendu wall < 20 min, `metrics.json` valide.
- [ ] P2.2 — `sbatch --account=aip-gdumas85_gpu scripts/slurm/smoke_sarl.sh breakout 6 42 50000 gpu` → attendu wall < 10 min.
- [ ] P2.3 — Comparer `metrics.json` CPU vs GPU sur `episode_returns[:20]` : divergence RNG attendue (outer loop non bit-exact GPU↔CPU, déjà documenté). Stats globales doivent rester du même ordre (pas de NaN, return_mean cohérent avec exploration pure pendant warmup).

**2.B — Benchmark (calibre les tranches SLURM)**
- [ ] P2.4 — `sbatch bench_sarl.sh breakout 6 42 500000 {cpu_4c,gpu_full}` (+ `gpu_mig` si Tamia l'a) → 2 ou 3 jobs, écrit `bench-*.json`.
- [ ] P2.5 — Analyse :
    - FPS / updates per second par mode
    - Extrapolation × 10 vers 5M frames → temps par cellule
    - Peak VRAM (pour dimensionner `--mem` et choisir MIG vs GPU complet)
    - Ratio GPU/CPU — si < 1.5× → rester CPU (queue CPU moins contentionnée même sur aip-*)
- [ ] P2.6 — **Décision partition** (écrite dans `docs/reports/sprint-07-profile.md`) :
    - Si `FPS_gpu_mig ≥ 0.6 × FPS_gpu_full` ET MIG dispo → **MIG** (concurrence max).
    - Sinon si `FPS_gpu_full ≥ 1.5 × FPS_cpu_4c` → **H100 full** (`--gres=gpu:h100:1`).
    - Sinon → **CPU**, `--cpus-per-task=4`.
- [ ] P2.7 — Mettre à jour `sarl_array.sh` : `--time=ceil(wall_500k × 11)` (10% marge), `--array=0-299%N` avec N = quota aip-* concurrent (à lire de `sshare`/test-only).

**2.C — Profil (identifie les pistes d'optim)**
- [ ] P2.8 — `uv run python -m cProfile -o prof.out scripts/run_sarl.py --game breakout --setting 6 --seed 42 --num-frames 100000 -o device=cpu` (CPU = profilable sans biais async GPU).
- [ ] P2.9 — Écrire `scripts/analyze_profile.py` (10 lignes) : `pstats` sort cumulative, print top 40 → stdout + `docs/reports/sprint-07-profile.md`.
- [ ] P2.10 — Refaire le profil sur 500 k frames pour voir l'effet post-warmup (ε a convergé, mix exploration/exploitation différent).
- [ ] P2.11 — Profil GPU séparé avec `torch.profiler` (sur 10 k frames, `device=cuda`) → ratio CPU-bound (env.step, tensor shuffling) vs GPU-bound (forward/backward). Rapport consigné.
- [ ] P2.12 — Croiser avec mon tableau d'hypothèses Phase 3 : valider / invalider / chiffrer chaque piste.

**Deliverables :**
- `outputs/bench/bench-{cpu_4c,gpu_full[,gpu_mig]}.json`
- `docs/reports/sprint-07-profile.md` : top 40 fonctions, décision partition, scoring pistes
- Tranches SLURM calibrées dans `sarl_array.sh`

**DoD :**
- Smoke CPU + GPU terminés sans erreur.
- ≥ 2 bench terminés, `bench-*.json` cohérents (seed-dup à ±5%).
- Rapport profil écrit avec top 5 bottlenecks mesurés.
- Décision partition actée.

**Risques :**
- **R2-1** Queue aip-* plus chargée qu'attendu pendant les tests → fallback Narval pour les smoke (coût : double setup, déjà mitigé par Phase 1 paths HPC générique).
- **R2-2** GPU ≈ CPU sur MinAtar → *pas un échec* mais implique de documenter et de basculer CPU proprement.
- **R2-3** `torch.profiler` sur fichiers CUDA peut planter sur H100 avec certaines versions PyTorch → fallback `cProfile` + `nvidia-smi dmon` manuel.

**Effort :** 3 h active + 1–4 h queue (non bloquant — autre travail en parallèle).

---

### Phase 3 — Optimisations parity-safe (0.5 j, **conditionnel**)

**But :** appliquer **uniquement** les optims à **risque parité zéro** validées par le profil Phase 2. Décision 2026-04-18 : *pas de dérivation numérique tolérée*, pas même à quelques ulps. Toutes les optims à risque > zéro (`torch.compile`, replay buffer tensor-backed, amp FP16/BF16) sont **retirées définitivement du périmètre Sprint 07**.

#### Pré-requis OBLIGATOIRE si GPU retenu (Phase 2 décide)

Avant toute exécution GPU, ajouter **une seule fois** en tête de `scripts/run_sarl.py` (et `run_sarl_cl.py`) :

```python
import torch
torch.set_float32_matmul_precision("highest")      # désactive TF32 matmul (H100 par défaut TF32)
torch.backends.cuda.matmul.allow_tf32 = False       # ceinture/bretelles
torch.backends.cudnn.allow_tf32 = False             # TF32 conv off
torch.backends.cudnn.deterministic = True           # reproductibilité conv
torch.backends.cudnn.benchmark = False              # pas de sélection dynamique de kernels
```

Et un test dédié `tests/unit/utils/test_fp32_strict.py` qui vérifie l'état de ces flags après `set_all_seeds()`. Si un flag est mal positionné, le run s'arrête.

Rationale : H100 active TF32 par défaut pour les matmul (10 bits de mantisse au lieu de 23), ce qui introduit une dérive ~10⁻³ incompatible avec l'objectif "zéro dérivation" du sprint.

#### Tableau d'optims retenues (toutes zéro-risque parité)

| # | Piste | Risque parité | Impact attendu | Effort | Condition d'application |
|---|-------|--------------|----------------|--------|-------------------------|
| O-1 | Vectoriser `target_wager` (`sarl/data.py:93-118`) : boucle Python batch-wise → récurrence vectorielle | **Zéro** (math scalaire identique à EMA séquentielle — à prouver par test diff) | 1–3 % wall | S (1 h + test diff atol=0) | Auto si profil confirme |
| O-2 | `non_terminal_idx` (`sarl/trainer.py:167`) : list-comp → `(is_terminal.squeeze() == 0).nonzero(as_tuple=True)[0]` | **Zéro** (ordre préservé) | < 1 % | XS (10 min) | Auto |
| O-3 | `get_state` (`sarl/data.py:82-90`) : prealloc buffer persistant côté trainer + `buf.copy_()` (au lieu de `torch.tensor(...).permute().unsqueeze().float().to()` à chaque frame) | **Zéro** | 2–5 % CPU, < 2 % GPU | S (2 h) | Auto si profil confirme |

**Règles de merge :**
- 1 commit par optim : `perf(sarl): <desc> — +X% wall (profile-cited, plan: docs/plans/plan-20260418-sprint07-sarl-tamia.md)`.
- `uv run pytest -q tests/parity/sarl` doit passer AVANT merge de chaque commit.
- **En plus** : un test `tests/parity/sarl/test_optim_bitexact.py` compare output avant/après optim sur 1000 samples synthétiques, `atol=0 rtol=0` (exact match).
- Si un test dévie d'un ulp : rollback immédiat. Pas de tolérance.

**Tâches :**
- [ ] P3.0 — (si GPU) Ajouter le bloc FP32-strict + son test unitaire. Commit `feat(sarl): enforce strict FP32 on H100 (no TF32 drift)`.
- [ ] P3.1 — Appliquer O-1, O-2, O-3 (auto si profil Phase 2 valide leur impact).
- [ ] P3.2 — Re-bench après optims : comparer `bench-post.json` vs `bench-pre.json` dans le rapport profil.

**Deliverables :**
- 1 à 4 commits sur `repro/sarl` (P3.0 + jusqu'à 3 × P3.1).
- Section "Optims appliquées" dans `sprint-07-profile.md` avec FPS avant/après et diff bit-exact.

**DoD :**
- Parité SARL Tier 1/2/3 verte après optims.
- `tests/parity/sarl/test_optim_bitexact.py` vert (atol=0).
- FP32-strict test vert si GPU.
- FPS post ≥ FPS pre (non-régression).

**Risques :**
- **R3-1** `torch.cumsum` sur l'EMA récursive de `target_wager` peut donner un ulp de différence si implémenté naïvement. Mitigation : tester d'abord à `atol=0` sur 10k samples ; si échec, revenir à la boucle Python (perte de ~2% wall, acceptable).
- **R3-2** `torch.backends.cudnn.deterministic=True` peut réduire la perf GPU de 5-15% (kernel déterministe parfois plus lent). *C'est un coût accepté* pour satisfaire la contrainte zéro dérivation.

**Effort :** 3–4 h.

---

### Phase 4 — Run l'array 300 cells (4 h–4 j wall-clock)

**But :** exécuter la grille complète.

- [ ] P4.1 — `sbatch --account=aip-gdumas85_{cpu|gpu} scripts/slurm/sarl_array.sh` (mode décidé Phase 2).
- [ ] P4.2 — Noter `ARRAY_JOB_ID`. `sbatch --dependency=afterany:<ID> scripts/slurm/aggregate.sh` (`afterany`, pas `afterok` — agréger même si cellules mortes).
- [ ] P4.3 — Monitoring : `bash scripts/slurm/monitor.sh` toutes les 2–4 h. Estimation terminaison via `squeue -o "%M %L"`.
- [ ] P4.4 — Spot-check mi-array : choisir 1 cellule en cours, comparer son FPS instantané (log `ep=X frames=Y`) à la fourchette Phase 2. Tolérance ±15 %.
- [ ] P4.5 — Cellules foirées : log `outputs/reports/failed_cells.md` avec `(game, setting, seed, exit_code, cause, log_path)`. **Pas de re-run silencieux** — décision Rémy avant toute resoumission.
- [ ] P4.6 — Budget cutoff : si wall-clock Phase 4 > 48 h après soumission → check queue saturation aip-*, check pré-emption, envisager Narval def-gpu en relais pour les cellules restantes.

**Deliverables :**
- 300 × `$SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<SEED>/metrics.json` (seeds 42-51)
- `outputs/reports/failed_cells.md`

**DoD :**
- ≥ 95 % des 300 cellules ont produit un `metrics.json` valide (`final_100_return` numeric, `total_frames ≥ 0.99 × num_frames`).
- Cellules échouées : cause identifiée pour chacune.
- `aggregate.sh` lancé (avec ou sans trous).

**Risques :**
- **R4-1** Pré-emption (même sur aip-*, rare mais possible) → `--requeue` dans sarl_array.sh.
- **R4-2** OOM : `--mem=8G` validé en Phase 2 mais surveiller le premier tranche d'array. `sacct -j <JOB_ID> --format=MaxRSS`.
- **R4-3** Dépassement `--time` sur 1–5 % des cellules (variance horizon d'épisodes MinAtar, surtout Freeway). Si > 10 %, re-calibrer.
- **R4-4** FS saturation sur `$SCRATCH` — 300 × (metrics.json ~200 KB + 2 × model.pt ~5 MB) ≈ 3 GB, négligeable, mais vérifier quota `quota -u`.

**Effort :** 3 h active + wall-clock selon Phase 2.

---

### Phase 5 — Aggregation + rapport (0.5 j)

**But :** produire le livrable scientifique.

- [ ] P5.1 — `aggregate.sh` complète : rsync `$SCRATCH` → `outputs/sarl/`, puis `scripts/aggregate_sarl.py` → `outputs/reports/sarl_summary.{json,md}`.
- [ ] P5.2 — Update `docs/reproduction/experiment_matrix.md` avec mean ± std + z par env.
- [ ] P5.3 — Écrire `docs/reports/sprint-07-report.md` :
    - Per-env table : our mean/std/z vs paper mean/std/z, verdict ±2σ
    - GPU-hours / CPU-hours utilisés, SLURM job IDs (provenance)
    - Renvoi vers `sprint-07-profile.md` pour optims
    - Section "Cellules échouées" si applicable
    - Section "Quirks Tamia" si on a dû adapter (TF32, H100, etc.)
- [ ] P5.4 — Si 1+ cellules hors ±2σ : section "Sprint 07 reproduction gaps" dans `docs/reproduction/deviations.md`. **Pas de tuning hyperparam pour combler le gap.**
- [ ] P5.5 — Update `docs/sprints/README.md` : Sprint 07 ✅ ou 🟡 (RG-00X si gaps).
- [ ] P5.6 — PR sur GitHub : `repro/sarl` → `main`.

**Deliverables :**
- `docs/reports/sprint-07-report.md`
- `outputs/reports/sarl_summary.{json,md}`
- Matrix + deviations à jour
- PR ouverte

**DoD :**
- Rapport sur main après merge PR.
- ≥ 2 envs MinAtar reproduisent à ±2σ (critère du sprint doc).
- Cellules manquantes documentées.
- `sprints/README.md` reflète le nouvel état.

---

## Risks & Unknowns (synthèse)

- **R-global-1** Tamia/aip-gdumas85 pas encore activé (Guillaume vient d'ajouter) → Phase 0 P0.3 le confirme.
- **R-global-2** Parité CPU-only ; H100 jamais testé sur ce code → Phase 0 P0.8 + Phase 2 P2.3 valident.
- **R-global-3** Branche perf-notes inaccessible (pas sur origin) → profil Phase 2 comme seule source de vérité. Acté.
- **R-global-4** N=10 seeds verrouillé (conforme papier). Si variance observée > paper std, c'est un résultat à rapporter, pas un prétexte pour ajouter des seeds (éviter le p-hacking). À noter en Phase 5.

## Verification Plan

- [ ] Phase 0 : `pytest tests/unit tests/parity` vert + smoke CPU `metrics.json` OK sur Tamia.
- [ ] Phase 1 : `sbatch --test-only` sur tous les sbatch, `test_paths.py` étendu passe.
- [ ] Phase 2 : `bench-*.json` cohérents (écart seed-dup < 5 %), décision partition loggée.
- [ ] Phase 3 : `tests/parity/sarl/*` vert après chaque commit `perf(sarl):`.
- [ ] Phase 4 : spot-check mi-array, FPS dans fourchette ±15 %.
- [ ] Phase 5 : `aggregate_sarl.py` exit 0 OU exit 1 avec liste explicite des trous.

## Definition of Done (plan global)

- [ ] Phase 0 : env Tamia opérationnel, repo cloné, tests verts.
- [ ] Phase 1 : repo consolidé, scaffolding SLURM complet, paths HPC propres.
- [ ] Phase 2 : bench + profil écrits, partition décidée.
- [ ] Phase 3 : optims parity-safe appliquées ou justification explicite de n'en appliquer aucune.
- [ ] Phase 4 : ≥ 95 % des 300 cellules ont produit un metrics.json valide.
- [ ] Phase 5 : `sprint-07-report.md` mergé sur main, `experiment_matrix.md` à jour.
- [ ] Critère papier : ≥ 2 envs MinAtar à ±2σ.

---

## Décisions verrouillées (2026-04-18)

| # | Décision | Verdict | Conséquence dans le plan |
|---|----------|---------|-------------------------|
| 1 | Suppression legacy (`SARL/`, `SARL_CL/`, `MARL/MAPPO-ATTENTION/`, `requirements-legacy.txt`) | ✅ Go | Phase 1.A |
| 2 | Cluster cible Tamia, fallback Narval | ✅ Validé (Tamia first ; debug avant bascule, pas d'abandon immédiat) | Phase 0 primary, fallback documenté |
| 3 | `torch.compile` (O-5) | ❌ Refusé (aucune dérivation tolérée) | O-5 retiré Phase 3 |
| 4 | N seeds | ✅ **N=10** (conforme papier) — 300 cellules | Phase 4 array size |
| 5 | TF32 sur H100 | ❌ Refus strict — FP32 forcé | Phase 3 pré-requis GPU obligatoire |

**Supersede :** `docs/sprints/sprint-07-reproduction-sarl.md` qui disait N=5 — à aligner en Phase 5 lors du rapport de sprint.

---

## Changelog

- 2026-04-18 (init) — plan créé (Rémy + Claude), cible Tamia, fallback Narval.
- 2026-04-18 (décisions) — verrouillage des 5 décisions : N=10 seeds, FP32 strict H100, O-5/O-6/O-7 retirées du périmètre, legacy delete go, scope strict Sprint 07.
