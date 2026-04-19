# Plan: Review progressive + reproduction intégrale du papier MAPS

**Date:** 2026-04-19
**Auteur:** Rémy Ramadour + Claude
**Branche:** `repro/paper-faithful` (nouvelle, à partir de `main` après merge de `repro/sarl`)
**Sprint:** Sprint-08 (this plan supersedes Phase 3+ of the old Sprint-07 plan)
**Estimated complexity:** XL (multiples semaines, ~80 sous-phases, exécutées séquentiellement)
**Status:** brouillon — valider avant exécution. **NE PAS DEMARRER** tant que Rémy n'a pas signé.

---

## Problem Statement

Objectif non-négociable : **reproduire intégralement le papier MAPS (Vargas et al., TMLR submission)**. Le papier couvre 5 domaines (Blindsight, AGL, SARL, SARL+CL, MARL), 7 settings (1-6 MAPS factorial + setting 7 = ACB baseline), avec N=500 seeds pour les tâches perceptuelles et N=3 seeds pour les tâches RL.

Audit 2026-04-19 a révélé les écarts suivants entre le working tree et ce que le papier exige :
- **Code MARL absent** de `src/maps/` (jamais porté ; restauré dans `MARL/MAPPO-ATTENTIOAN/` depuis git).
- **Setting 7 (ACB) absent** de la factorielle courante (6 settings au lieu de 7).
- **Hyperparams SARL/SARL+CL divergent du papier Table 11** sur 5-6 axes (num_frames, lr_second_order, alpha EMA, gamma DQN, scheduler step, poids CL).
- **MeltingPot env non-installé.** Recipe existe (DeepMind open-source, dmlab2d Python 3.10 wheel) mais jamais exécutée.
- **Parité actuelle vise le code du student**, pas le papier. Décision Rémy 2026-04-19 : papier = source de vérité.
- **Storage Tamia non-discipliné** : `.venv/` 7.2 GB et tous les outputs sont en `/project`. Guillaume : "Tamia fait pour utiliser $SCRATCH quasi exclusivement, il n'y a pas grand chose en home/project".
- **Queue labo partagée** (Guillaume, Rémy, Nadine, MARL) : discipline chainée, ≤ 2 jobs actifs max.

## Scientific Rationale

**Pourquoi ce sprint et pas une reprise du Sprint-07 ?** Sprint-07 visait un objectif trop étroit (SARL seul, parité maps.py) avec des hypothèses erronées (GPU ~500 fps, N=10 seeds alignée papier). Les mesures Phase 2 du Sprint-07 ont montré que :
- GPU/CPU ratio est setting-dépendant (CPU bat GPU sur settings rapides, GPU gagne 2× sur cascade complète).
- Settings 5-6 ne rentrent pas dans 24h Tamia → il faut checkpoint/resume.
- `target_wager` (32% wall setting 1) et `torch.dropout` in cascade (31% wall setting 6) dominent les profils ; l'optim O-1 du Sprint-07 était sous-estimée d'un facteur 10×.

**Pourquoi papier > maps.py ?** Parce que l'objectif est de reproduire ce que dit **le papier**, pas ce que faisait le code du student. Quand ils divergent (et ils divergent), le papier doit gagner — sinon on ne reproduit pas le papier, on réplique le code. Les deux sont différents.

**Pourquoi composante par composante et pas un big-bang ?** Parce qu'on a déjà vu ce qu'un plan trop macro donne (Sprint-07 Phase 1 a supprimé MAPPO-ATTENTIOAN sans port existant → mandat de restauration). La granularité atomique (1 sous-phase = 1 commit) force :
- Chaque décision à être examinée isolément.
- Les tests de parité à passer vert à chaque étape (pas de régression cachée par d'autres changements).
- La rollback facile si un pas casse quelque chose en aval.

**Références méthodologiques :**
- Papier : `pdf/MAPS_TMLR_Journal_Submission.pdf` (30 p, lu intégralement 2026-04-19).
- Tables 9 (Blindsight), 10 (AGL), 11 (SARL/SARL+CL), 12 (MARL) = hyperparams papier.
- Tables 5 (Blindsight/AGL), 6 (SARL), 7 (MARL) = z-scores cibles pour la repro.
- Equations 1-18 = architectures et losses canoniques.

---

## Principes verrouillés (2026-04-19, non-négociables)

1. **Papier = source de vérité.** Quand code student et papier divergent → papier gagne. Toute divergence restante = entrée dans `docs/reproduction/deviations.md` avec référence équation/table + justification explicite.
2. **Composante par composante.** Pas de refactor "big bang". Chaque fichier = son propre review, son propre fix, son propre commit.
3. **Preserve everything.** Rien ne se supprime sans port vérifié (parité `atol=1e-6`). Référence dans `external/paper_reference/` reste en place.
4. **Plans execute sequentially.** Pas de flancs parallèles, pas de "X pendant que tu fais Y", pas de "sauter une étape parce que urgent". Une phase à la fois, DoD validée, puis suivante.
5. **Quality over speed.** Pas de réduction de scope préventive. Si c'est long, c'est long.
6. **Discipline cluster DRAC :**
   - Code reste dans `/project/6102289/rram17/Workspace/MAPS/MAPS_PROJECT` (instruction Rémy).
   - Outputs, logs, checkpoints intermédiaires → `$SCRATCH/maps/…`.
   - `.venv/` → `$SCRATCH/maps/venv/` avec symlink depuis le repo (instruction Guillaume).
   - ≤ 2 jobs actifs au total sur `aip-gdumas85`. Séries = `--dependency=afterany`.
7. **Config-first.** Tout paramètre modifiable via `config/*.yaml`. Magic numbers interdits en Python (hook `python-quality-check.sh` les flagge).
8. **Verification every step.** Pas de commit sans parité verte + checklist `verification-before-completion` passée.

---

## Data flow (vue globale)

```
Phase 0 — Setup (branche, tidy git)
    ↓
Phase A — Hygiène cluster (venv+outputs sur $SCRATCH ; helper sbatch chaînage)
    ↓
Phase B — Audit papier vs code (doc écrit, zéro code)
    ↓
Phase C — Review core MAPS components (cascade, second-order, losses, networks, utils)
    ↓
Phase D — Review per-experiment (SARL, SARL+CL, Blindsight, AGL)
    ↓
Phase E — Extensions (ACB setting 7, MeltingPot install, MARL port)
    ↓
Phase F — Runs reproduction (5 domaines × 7 settings × N seeds)
    ↓
Phase G — Rapport + merge main
```

A → G strictement séquentiel. Une sous-phase à la fois.

---

## Key components impactés

| Composante | Phases qui la touchent |
|---|---|
| `src/maps/components/cascade.py` | C.1, C.2 |
| `src/maps/components/second_order.py` | C.3-C.6 |
| `src/maps/components/losses.py` | C.7-C.9 |
| `src/maps/networks/first_order_mlp.py` | C.10, C.11 |
| `src/maps/utils/*` (5 fichiers) | C.12-C.16 |
| `src/maps/experiments/sarl/*` (7 fichiers) | D.1-D.14 |
| `src/maps/experiments/sarl_cl/*` (4 fichiers) | D.15-D.21 |
| `src/maps/experiments/blindsight/*` | D.22-D.24 |
| `src/maps/experiments/agl/*` | D.25-D.27 |
| `src/maps/experiments/sarl/acb.py` (nouveau) | E.1-E.5 |
| `src/maps/experiments/marl/*` (nouveau) | E.10-E.21 |
| `config/training/*.yaml` | B.10-B.11, D.11, D.19, E.17 |
| `config/paths.yaml` | A.3 |
| `docs/reproduction/*.md` | B, D, G |
| `docs/install_*.md` | A.7, E.6-E.9 |
| `scripts/slurm/*` | A.5, E.19, F.3-F.5 |
| `tests/parity/*` | C+D (updates), E.4, E.20 |

## Config changes needed

Les changements de config (tous après validation papier Phase B, appliqués en Phase D) :

```yaml
# config/training/sarl.yaml — Table 11 papier
training:
  num_frames: 500_000         # or 1_000_000 (paper table vs text — to decide Phase B)
optimizer:
  lr_second_order: 0.0002     # was 0.00005
# trainer.py GAMMA = 0.999    # was hardcoded 0.99
alpha: 45.0                   # was 1.0 (→ /100 = 0.01, paper wants 0.45)
scheduler:
  step_size: 1                # was 1000 (paper Table 11)
```

```yaml
# config/training/sarl_cl.yaml — Table 11 papier
cl:
  weight_task: 0.3            # was 1.0
  weight_distillation: 0.6    # was 1.0 (paper calls it "weight regularization")
  weight_feature: 0.1         # was 1.0
```

```yaml
# config/training/marl.yaml — NEW, Table 12 papier
hidden_size: 100
optimizer:
  actor_lr: 7e-5
  critic_lr: ...               # paper says "100" — likely typo; investigate Phase B
  weight_decay: 1e-5
ppo:
  epoch: 15
  clip_param: 0.2
  entropy_coef: 0.01
num_env_steps: 300_000         # text says 300k, table says 15e6 — to decide Phase B
```

## Risks & Unknowns

| ID | Item | Mitigation |
|---|---|---|
| R-1 | Paper Table 11 dit `num_frames=500k`, texte page 13 dit `1M`. Inconsistance papier. | Phase B.10 : trancher (convention TMLR = table appendix). Documenter dans deviations.md. |
| R-2 | Paper Table 11 dit `lr_second_order=0.0002` mais `maps.py` a `5e-5`. Student a peut-être trouvé un bug. | Phase B : essayer les 2 en parallèle sur 1 seed, comparer convergence. Trancher papier par défaut sauf contre-preuve forte. |
| R-3 | MeltingPot upstream a bougé depuis paper submission (mai 2025). API break probable. | Phase E.8 : pinner un commit ≤ 2025-05. Si rien ne marche, mailer DeepMind support. |
| R-4 | dmlab2d n'a pas de wheel Python 3.12 | Phase E.6 : venv Python 3.10 séparé pour MARL, co-existence OK sur `$SCRATCH`. |
| R-5 | Port MARL est long (MAPPO + GRU + positional encoding + second-order + cascade). | Pas de raccourcis. Découper en 10+ sous-phases (E.10-E.21). |
| R-6 | CAE_loss vs contrastive (eq.4 paper SimCLR) — grosse divergence sémantique. | Phase C.7-C.9 : dédier du temps, décider avec Rémy. Probable implémentation paper-faithful en paralèle du CAE historique (flag mode). |
| R-7 | Reprod échoue pour 1 domaine malgré alignement papier | Honnête dans le rapport final (Phase G). Résultat scientifique valide : "la méthodo publiée ne suffit pas à reproduire X". |
| R-8 | Compute Tamia saturé par un autre labo | Fallback Narval `def-gdumas85` (déjà documenté Sprint-07). |
| R-9 | `target_wager` Python loop batch-wise (profile Sprint-07 Phase 2 : 32% wall setting 1) | Phase D.2 : vectoriser ; parité bit-exact sur 10k samples avant merge. |
| R-10 | Dropout in cascade (31% wall setting 6) — retrait a risque parité | Phase D.4 : étudier si dropout est load-bearing dans la cascade inner loop, discuter avec Rémy. |

---

## Verification Plan (global)

Chaque sous-phase spécifie sa propre vérification. À l'échelle du plan, les vérifications globales sont :

- [ ] `pytest tests/unit tests/parity` vert à chaque commit.
- [ ] `uv run ruff check .` et `ruff format --check` vert à chaque commit.
- [ ] `python-quality-check.sh` hook ne flagge aucun avertissement bloquant.
- [ ] `verification-before-completion` checklist (Tier 1) passée à chaque commit.
- [ ] Chaque entrée de `deviations.md` a référence équation/table papier + justification.
- [ ] Chaque fichier touché cite ses sources papier en docstring.
- [ ] À la fin de F : ≥ 4/5 domaines reproduits à ±2σ des valeurs papier.

## Definition of Done (global)

- [ ] Toutes les sous-phases A.* → G.* complétées, chacune avec son commit et son DoD.
- [ ] Tous les 5 domaines + 7 settings exécutables (smoke run clean sur Tamia / Mac).
- [ ] Tous les runs F.* aggrégés dans `outputs/reports/paper-reproduction-*`.
- [ ] Rapport final `docs/reports/paper-reproduction-report.md` mergé sur `main`.
- [ ] Branche `repro/paper-faithful` mergée sur `main` (PR).
- [ ] `experiment_matrix.md` à jour avec valeurs mesurées finales.
- [ ] `deviations.md` exhaustif et finalisé.

---

# Phases détaillées

Chaque sous-phase a : **titre · préambule court · ce qu'on fait · vérification · DoD · effort estimé · dépendance**. Dépendance = sous-phase précédente sauf indication contraire. Ordre strict.

## Phase 0 — Setup branche (1 h)

### 0.1 Valider + merger `repro/sarl` vers `main`
- **Faire :** PR / merge `repro/sarl` → `main`. Les 4 commits récents (scratch_root, scaffolding SLURM, phase 2 bench report, restauration MAPPO) sont suffisamment propres pour rejoindre `main`. Décision Rémy : "merge si tu veux".
- **Vérifier :** `git log main --oneline -5` montre les 4 commits ; `git status` clean.
- **DoD :** `main` contient la restauration MAPPO + les fixes Sprint-07 Phase 1-2 ; pas de branche legacy orpheline.
- **Effort :** 10 min + review.

### 0.2 Créer branche `repro/paper-faithful` depuis `main`
- **Faire :** `git checkout main && git checkout -b repro/paper-faithful && git push -u origin repro/paper-faithful`.
- **Vérifier :** branche visible sur origin.
- **DoD :** working tree sur `repro/paper-faithful`, clean, à jour.
- **Effort :** 2 min.

---

## Phase A — Hygiène cluster (prérequis Tamia) (0.5-1 j)

**But :** être en conformité storage avant toute nouvelle exécution. Sinon risque kick DRAC.

### A.1 Créer la structure `$SCRATCH/maps/`
- **Faire :** `mkdir -p $SCRATCH/maps/{venv,outputs,logs,checkpoints,bench,reports}`. Documenter la structure attendue dans `docs/install_tamia.md`.
- **Vérifier :** `ls -la $SCRATCH/maps/` montre 6 sous-dirs.
- **DoD :** hiérarchie scratch prête.
- **Effort :** 2 min.

### A.2 Déplacer `.venv/` vers `$SCRATCH/maps/venv/` + symlink
- **Faire :**
  ```bash
  mv .venv $SCRATCH/maps/venv
  ln -s $SCRATCH/maps/venv .venv
  ```
- **Vérifier :** `ls -la .venv` = symlink ; `uv run --offline python -c "import torch; print(torch.__version__)"` fonctionne (via login + via sbatch compute).
- **DoD :** venv 7.2 GB hors `/project`, symlink en place, `uv run --offline` OK depuis login et compute.
- **Effort :** 15 min (copie + vérif depuis sbatch test).

### A.3 Ajouter `scratch_root` aux trainers (écriture réelle)
- **Préambule :** Sprint-07 P1.4 a ajouté le champ `paths.scratch_root` mais aucun trainer ne l'utilise ; on écrit encore sous `paths.outputs` (= project tree).
- **Faire :** modifier `src/maps/experiments/{sarl,sarl_cl,blindsight,agl}/training_loop.py` : le `_persist_outputs()` (ou équivalent) écrit sous `paths.scratch_root / "maps/outputs/<domain>/..."` au lieu de `paths.outputs`.
- **Vérifier :** smoke `run_sarl.py ... --num-frames 10000 -o device=cpu` écrit `metrics.json` sous `$SCRATCH/maps/outputs/sarl/...` ; plus sous `/project/…/outputs/`.
- **DoD :** 4 trainers mis à jour ; smoke tests passent.
- **Effort :** 30 min.

### A.4 Helper `scripts/slurm/submit_chained.sh`
- **Faire :** nouveau script `scripts/slurm/submit_chained.sh` qui prend une liste de scripts sbatch et les chaîne via `--dependency=afterany:<prev_id>`. Usage `submit_chained.sh bench_cpu.sh bench_gpu.sh profile.sh`.
- **Vérifier :** `sbatch --test-only` sur une série de 3 scripts parse OK ; dépendances correctement ajoutées.
- **DoD :** script + doc d'usage dans header.
- **Effort :** 30 min.

### A.5 Mettre à jour `scripts/slurm/*.sh` pour utiliser `$SCRATCH` par défaut
- **Faire :** `smoke_sarl.sh`, `bench_sarl.sh`, `sarl_array.sh`, `profile_sarl.sh`, `aggregate.sh` utilisent déjà `$SCRATCH` (Sprint-07) mais vérifier qu'ils pointent bien vers `$SCRATCH/maps/…` pas `$SCRATCH/…` direct (cohérence avec A.1).
- **Vérifier :** 5 × `sbatch --test-only` OK.
- **DoD :** tous les sbatch pointent `$SCRATCH/maps/…`.
- **Effort :** 15 min.

### A.6 Mettre à jour `docs/install_tamia.md`
- **Faire :** ajouter 3 sections : (1) "venv on $SCRATCH + symlink", (2) "shared queue discipline ≤ 2 jobs, --dependency=afterany par défaut", (3) "VSCode interdit sur login (wiki TamIA)". Inclure commandes exactes + policy.
- **Vérifier :** doc lu + validé par Rémy.
- **DoD :** `docs/install_tamia.md` reflète la politique 2026-04-19.
- **Effort :** 30 min.

### A.7 Commit Phase A
- **Faire :** `chore(cluster): enforce $SCRATCH-centric storage + chained-job helper (Phase A)`.
- **Vérifier :** hook `git-discipline.sh` ne flagge rien de bloquant ; `pytest tests/unit` vert.
- **DoD :** commit sur `repro/paper-faithful`.
- **Effort :** 5 min.

---

## Phase B — Audit papier vs code (1-2 j, zéro code)

**But :** produire la vérité écrite "papier ↔ monolithe student ↔ port courant" avant toute modification. Rien de code touché dans cette phase.

### B.1 Extraire Table 11 du papier (hyperparams SARL/SARL+CL)
- **Faire :** copier Table 11 du papier dans un nouveau `docs/reproduction/paper_tables_extracted.md` section "Table 11 — MinAtar hyperparams". Cite exacte + référence page.
- **Vérifier :** 23 hyperparams retranscrits, zéro paraphrase (paper Table 11 p. 30 = 23 rows).
- **DoD :** doc existe, Table 11 copiée intégralement.
- **Effort :** 15 min.

### B.2 Extraire Table 9 du papier (Blindsight hyperparams)
- **Faire :** idem Table 9 dans `paper_tables_extracted.md`.
- **Vérifier :** 11 hyperparams retranscrits.
- **DoD :** Table 9 présente.
- **Effort :** 10 min.

### B.3 Extraire Table 10 du papier (AGL hyperparams)
- **Faire :** idem Table 10.
- **Vérifier :** 13 hyperparams.
- **DoD :** Table 10 présente.
- **Effort :** 10 min.

### B.4 Extraire Table 12 du papier (MeltingPot hyperparams)
- **Faire :** idem Table 12. Flagger les incohérences (critic_lr=100 est suspect, num_env_steps=15e6 vs texte 300k).
- **Vérifier :** 12 hyperparams + flags.
- **DoD :** Table 12 présente.
- **Effort :** 10 min.

### B.5 Extraire équations 1-18 du papier
- **Faire :** LaTeX-source des 18 équations copiées dans `paper_equations_extracted.md` avec leur contexte (page, description).
- **Vérifier :** 18 équations + contexte.
- **DoD :** doc existe.
- **Effort :** 20 min.

### B.6 Extraire Tables 5/6/7 du papier (z-scores cibles)
- **Faire :** Tables 5 (Blindsight/AGL), 6 (SARL), 7 (MARL) copiées dans `paper_targets_extracted.md`.
- **Vérifier :** z-scores par setting × env pour chaque domaine.
- **DoD :** doc existe.
- **Effort :** 20 min.

### B.7 Audit croisé SARL : paper Table 11 ↔ `external/paper_reference/sarl_maps.py` ↔ `src/maps/experiments/sarl/` + `config/training/sarl.yaml`
- **Faire :** nouveau doc `docs/reproduction/paper_vs_code_audit.md` — section "SARL". Tableau : Hyperparam · Papier · Monolithe · Port+Config · ∆ · Action.
- **Vérifier :** ~25 lignes, au moins 5 divergences identifiées.
- **DoD :** tableau complet.
- **Effort :** 1.5 h.

### B.8 Audit croisé SARL+CL : paper ↔ `sarl_cl_maps.py` ↔ `src/maps/experiments/sarl_cl/`
- **Faire :** même méthode, section "SARL+CL" du même doc. Inclure poids CL (0.3/0.6/0.1 papier vs 1.0/1.0/1.0 notre config).
- **Vérifier :** divergences flagguées.
- **DoD :** section complète.
- **Effort :** 1 h.

### B.9 Audit croisé Blindsight : paper Table 9 ↔ `blindsight_tmlr.py` ↔ `src/maps/experiments/blindsight/`
- **Faire :** section "Blindsight".
- **Vérifier :** divergences identifiées ; lien avec RG-002.
- **DoD :** section complète.
- **Effort :** 1 h.

### B.10 Audit croisé AGL : paper Table 10 ↔ `agl_tmlr.py` ↔ `src/maps/experiments/agl/`
- **Faire :** section "AGL". Inclure question downstream training phase (RG-003).
- **Vérifier :** divergences + RG-003.
- **DoD :** section complète.
- **Effort :** 1 h.

### B.11 Audit croisé MARL : paper Table 12 ↔ `MARL/MAPPO-ATTENTIOAN/` (pas de port à comparer)
- **Faire :** section "MARL". Identifier structure modules MAPPO-ATTENTIOAN + mapping vers futur `src/maps/experiments/marl/`.
- **Vérifier :** map fonctionnelle documentée.
- **DoD :** section complète.
- **Effort :** 2 h.

### B.12 Consolider `docs/reproduction/deviations.md`
- **Faire :** ajouter toutes les divergences identifiées en B.7-B.11 comme entrées D-NNN (numérotation continuée depuis D-004 existant). Chaque entrée : location, paper says, code does, action proposée, rang de la phase qui la traitera.
- **Vérifier :** ≥ 15 nouvelles entrées (estimé 5-6 par domaine).
- **DoD :** `deviations.md` exhaustif.
- **Effort :** 1 h.

### B.13 Mettre à jour `docs/reproduction/experiment_matrix.md`
- **Faire :**
  - Corriger N=10 → N=3 pour SARL/SARL+CL/MARL (N=500 pour Blindsight/AGL).
  - Ajouter ligne setting 7 ACB par domaine.
  - Recopier z-scores papier Tables 5-7 comme cibles.
- **Vérifier :** matrice cohérente avec le papier.
- **DoD :** matrice à jour.
- **Effort :** 45 min.

### B.14 Commit Phase B
- **Faire :** `docs(repro): paper-vs-code audit + updated deviations and matrix (Phase B)`.
- **Vérifier :** aucun fichier code touché, seulement docs.
- **DoD :** commit sur `repro/paper-faithful`.
- **Effort :** 5 min.

---

## Phase C — Review core MAPS components (3-5 j)

**But :** relire les 5 composantes core `src/maps/` (cascade, second_order, losses, networks, utils) contre papier + monolithes restaurés. Un fichier = 1-3 sous-phases (review + fix + parité mise à jour).

### Méthode par composante

Chaque composante suit ce pattern :
1. **Review** (lecture ciblée + notes dans `docs/reviews/component-<name>.md`).
2. **Plan fix** (si écarts trouvés ; liste ordonnée).
3. **Apply fix** (commits atomiques, 1 par écart).
4. **Parity re-test** après chaque fix.

### C.1 Review `src/maps/components/cascade.py` (79 L)
- **Faire :** comparer `cascade_update` et `n_iterations_from_alpha` à paper eq.6 et à son usage dans `sarl_maps.py` + `blindsight_tmlr.py` + `agl_tmlr.py`. Écrire `docs/reviews/cascade.md` : (a) conformité eq.6, (b) α bounds, (c) cohérence API cross-domaine, (d) pistes optim (vectorisation de la boucle cascade inner).
- **Vérifier :** review relue par Rémy ; fixes identifiés.
- **DoD :** doc `docs/reviews/cascade.md` ; rien de code touché.
- **Effort :** 30 min.

### C.2 Apply fixes cascade (si besoin)
- **Faire :** commits atomiques par fix identifié en C.1.
- **Vérifier :** `pytest tests/unit/components/test_cascade.py tests/parity/sarl` vert.
- **DoD :** fixes landés, parité verte.
- **Effort :** variable (0 à 2 h selon trouvailles C.1).

### C.3 Review `src/maps/components/second_order.ComparatorMatrix`
- **Faire :** comparer à paper eq.1 (`C = X - Ŷ`). Vérifier shape-handling + detach.
- **Vérifier :** doc `docs/reviews/second_order.md` section "ComparatorMatrix".
- **DoD :** section complète.
- **Effort :** 20 min.

### C.4 Review `src/maps/components/second_order.WageringHead`
- **Faire :** comparer à paper eq.2-3 (`C' = Dropout(C)` puis `W = WC' + b`). Vérifier : dropout p=0.1 paper, linear out 2 units raw logits, weight init uniforme (0, 0.1).
- **Vérifier :** section "WageringHead" du même doc.
- **DoD :** section complète.
- **Effort :** 20 min.

### C.5 Review `src/maps/components/second_order.SecondOrderNetwork`
- **Faire :** composition ComparatorMatrix + WageringHead + cascade_update. Vérifier forward signature + cascade_rate passage.
- **Vérifier :** section "SecondOrderNetwork".
- **DoD :** section complète.
- **Effort :** 20 min.

### C.6 Apply fixes second_order (si besoin)
- **Faire :** commits atomiques.
- **Vérifier :** `pytest tests/unit/components/test_second_order.py tests/parity/sarl` vert.
- **DoD :** parité verte.
- **Effort :** variable.

### C.7 Review `src/maps/components/losses.cae_loss`
- **Faire :** comparer à paper eq.4 (contrastive SimCLR-like). C'est une divergence majeure connue (D-002). Lister les deux variants : CAE (Rifai 2011) = notre code, contrastive (Chen 2020 SimCLR) = papier. Documenter dans `docs/reviews/losses.md` : équivalence ou pas ?
- **Vérifier :** review écrite.
- **DoD :** section "cae_loss vs contrastive".
- **Effort :** 1.5 h (lecture dense).

### C.8 Décision CAE-vs-contrastive (requires Rémy)
- **Faire :** présenter à Rémy : (a) garder CAE par défaut + ajouter contrastive en option, (b) remplacer CAE par contrastive (paper-faithful), (c) autre. Wait go.
- **Vérifier :** décision écrite dans `deviations.md`.
- **DoD :** décision validée.
- **Effort :** 10 min review + attente Rémy.

### C.9 Implémenter la décision C.8
- **Faire :** selon (a/b/c), implémenter. Si (b), casser la parité Tier 1/2/3 (pointant maps.py qui fait CAE) → refactor les `_reference_sarl.py` slices vers paper-faithful ET renommer les tests historiques `test_tier*_historical.py` (archivés).
- **Vérifier :** toute la suite pytest verte dans le nouveau mode ; ancienne parité archivée.
- **DoD :** décision appliquée, suite verte.
- **Effort :** 1-4 h selon décision.

### C.10 Review autres losses (`components/losses.py`)
- **Faire :** `distillation_loss`, autres helpers. Comparer à paper §Continual Learning (eq.15-18).
- **Vérifier :** section du doc `docs/reviews/losses.md`.
- **DoD :** section complète.
- **Effort :** 30 min.

### C.11 Review `src/maps/networks/first_order_mlp.py` (121 L)
- **Faire :** comparer à BLINDSIGHT/AGL monoliths (Blindsight §2.2 fig 2). Vérifier hidden_dim (Blindsight=100, AGL=40 per Table 9-10), chunked-sigmoid AGL (D-004), decoder_activation.
- **Vérifier :** section `docs/reviews/first_order_mlp.md`.
- **DoD :** doc complet.
- **Effort :** 45 min.

### C.12 Apply fixes first_order_mlp (si besoin)
- **Faire :** commits atomiques.
- **Vérifier :** `pytest tests/unit/networks/ tests/parity/_reference_{agl,blindsight}*` vert.
- **DoD :** parité verte.
- **Effort :** variable.

### C.13 Review `src/maps/utils/seeding.py`
- **Faire :** vérifier que `set_all_seeds(seed)` couvre : `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic`. Comparer au papier ("a dedicated np.random.RandomState(seed) for deterministic NumPy operations").
- **Vérifier :** section `docs/reviews/utils.md` → seeding.
- **DoD :** seeding vérifié ; test ajouté si manquant.
- **Effort :** 30 min.

### C.14 Review `src/maps/utils/config.py`
- **Faire :** vérifier OmegaConf wrapper, override handling. Pas critique parité.
- **Vérifier :** section utils.md.
- **DoD :** section.
- **Effort :** 20 min.

### C.15 Review `src/maps/utils/paths.py`
- **Faire :** déjà touché A.3 ; re-vérifier cohérence + couverture scratch_root.
- **Vérifier :** section utils.md.
- **DoD :** section.
- **Effort :** 15 min.

### C.16 Review `src/maps/utils/logging_setup.py`
- **Faire :** conformité `logging` (pas `print`), formatting cohérent avec stdlib.
- **Vérifier :** section utils.md.
- **DoD :** section.
- **Effort :** 20 min.

### C.17 Review `src/maps/utils/energy_tracker.py`
- **Faire :** comparer au papier §3 : "we used nvidia-smi through Python's subprocess module to query GPU power consumption. The carbon footprint was calculated using the energy usage as well as the country specific value of kg of CO2 / kWh (Canada is 0.1 kg of CO2 / kWh)".
- **Vérifier :** section utils.md + constante 0.1 kg CO2/kWh présente.
- **DoD :** section.
- **Effort :** 30 min.

### C.18 Commit Phase C
- **Faire :** si commits atomiques déjà faits en C.2, C.6, C.9, C.12 → juste vérifier suite complète ; sinon commit final "review(core): component audits + fixes (Phase C)".
- **Vérifier :** `pytest tests/` full suite verte ; `ruff check` + `ruff format --check` vert.
- **DoD :** Phase C clôturée.
- **Effort :** 15 min.

---

## Phase D — Review per-experiment (5-10 j)

**But :** review fichier-par-fichier pour chacun des 4 experiments portés. Plus les extensions checkpoint/resume + alignement config papier.

### Méthode (identique à Phase C)
1. Review (doc `docs/reviews/experiment-<name>-<file>.md`).
2. Fixes (commits atomiques).
3. Parité verte.

### SARL (7 fichiers à relire)

### D.1 Review `src/maps/experiments/sarl/data.py`
- **Faire :** comparer à `external/paper_reference/sarl_maps.py` (replay_buffer, transition namedtuple, get_state, target_wager). Vérifier alpha/100 convention, boucle Python dans `target_wager` (hotspot profile Sprint-07 : 32% wall setting 1).
- **Vérifier :** doc `docs/reviews/sarl-data.md`.
- **DoD :** doc complet + liste fixes priorisée.
- **Effort :** 1 h.

### D.2 Fix `target_wager` vectorisation + alpha alignment
- **Faire :**
  - Vectoriser `target_wager` : la boucle Python → `torch.zeros` + cumulative EMA vectorisé. Preserver sémantique scalaire exacte (test bit-exact sur 10k samples synthetic avant merge).
  - Aligner `alpha=45.0` dans `config/training/sarl.yaml` (→ `/100` = 0.45 papier).
  - Documenter la divergence papier (`alpha=0.45` direct) vs convention student (`alpha=45 /100`) dans `deviations.md`.
- **Vérifier :** `tests/parity/sarl` vert ; nouveau test `tests/unit/experiments/sarl/test_target_wager_vectorized.py` bit-exact contre loop reference.
- **DoD :** commit `perf(sarl): vectorize target_wager + align alpha to paper (D-NNN)`.
- **Effort :** 2 h.

### D.3 Review `src/maps/experiments/sarl/model.py`
- **Faire :** comparer `SarlQNetwork` et `SarlSecondOrderNetwork` à paper eq.7-12 + paper fig.3. Vérifier tied-weights reconstruction (`F.linear(hidden, fc.weight.t())`), dropout in cascade loop (hotspot profile : 31% wall setting 6).
- **Vérifier :** doc `docs/reviews/sarl-model.md`.
- **DoD :** doc complet.
- **Effort :** 1 h.

### D.4 Fix dropout-in-cascade (si décision parity-safe)
- **Préambule :** dropout est dans le forward de `SarlSecondOrderNetwork`, donc appelé 50× par update en setting 6 (cascade 50 iter). Question : est-ce que dropout dans la cascade inner loop est intentionnel (le papier eq.2 met dropout **avant** la cascade, cf. "`C' = Dropout(C)`" puis cascade inside `W_t`) ?
- **Faire :** relecture eq.2-3 minutieuse. Si dropout ne doit s'appliquer qu'une fois (avant la cascade, pas à chaque itération), le sortir de la boucle. Gain perf significatif, doit rester parity-safe vs papier (mais cassera parité maps.py — flagger).
- **Vérifier :** tests parity mis à jour ; bench setting 6 refait → fps mesurée.
- **DoD :** commit + bench avant/après dans PR desc.
- **Effort :** 3 h (discussion + impl + re-bench).

### D.5 Review `src/maps/experiments/sarl/rollout.py`
- **Faire :** `epsilon_greedy_action`, `greedy_action`, `anneal_epsilon`. Comparer à paper (ε annealing schedule).
- **Vérifier :** doc `docs/reviews/sarl-rollout.md`.
- **DoD :** doc complet.
- **Effort :** 30 min.

### D.6 Review `src/maps/experiments/sarl/trainer.py`
- **Faire :** `sarl_update_step`. Vérifier :
  - `GAMMA=0.99` hardcoded ligne 54 vs papier `0.999` Table 11.
  - `non_terminal_idx` list-comp (plan O-2 du Sprint-07).
  - Ordre backward/step dans la branche meta (load-bearing per Sprint-04b).
  - Cascade loop Python (plan O-B du profile).
- **Vérifier :** doc `docs/reviews/sarl-trainer.md`.
- **DoD :** doc complet + priorités fixes.
- **Effort :** 1 h.

### D.7 Fix GAMMA config-driven + non_terminal_idx optimization
- **Faire :**
  - Sortir `GAMMA` hardcoded → config `training.gamma`. Default paper = 0.999.
  - Remplacer `non_terminal_idx = torch.tensor([i for i, done in enumerate(is_terminal) if done == 0], …)` par `torch.nonzero(is_terminal.squeeze() == 0, as_tuple=True)[0]`.
- **Vérifier :** parité bit-exact sur 10k samples ; `pytest tests/parity/sarl/test_tier3_update.py` vert.
- **DoD :** commit `fix(sarl): gamma config-driven + non_terminal_idx vectorized (D-NNN)`.
- **Effort :** 1 h.

### D.8 Review `src/maps/experiments/sarl/training_loop.py`
- **Faire :** `run_training`, `setting_to_config`, `_build_networks`, `_build_optimizers`, `_persist_outputs`. Vérifier : optimizer Adam + eps (table dit `minimum_squared_gradient=0.01`, config dit `eps: 0.01` OK), scheduler StepLR step_size 1 (papier) vs 1000 (config).
- **Vérifier :** doc `docs/reviews/sarl-training-loop.md`.
- **DoD :** doc complet.
- **Effort :** 1.5 h.

### D.9 Fix scheduler step_size + lr_second_order
- **Faire :**
  - `config.scheduler.step_size`: 1000 → 1 (paper Table 11).
  - `config.optimizer.lr_second_order`: 5e-5 → 2e-4 (paper Table 11).
  - Documenter divergences dans `deviations.md`.
- **Vérifier :** smoke run passe ; observer impact sur learning curves (sera re-mesuré Phase F).
- **DoD :** commit `fix(sarl): align scheduler step_size + lr_2 to paper Table 11 (D-NNN)`.
- **Effort :** 30 min.

### D.10 Review `src/maps/experiments/sarl/evaluate.py`
- **Faire :** `run_greedy_episode`, `aggregate_validation`. Pas critical parité.
- **Vérifier :** doc `docs/reviews/sarl-evaluate.md`.
- **DoD :** doc.
- **Effort :** 30 min.

### D.11 Review `src/maps/experiments/sarl/losses.py`
- **Faire :** `cae_loss` (déjà traité C.7-C.9 ? ou duplicate). Clarifier rôle.
- **Vérifier :** doc.
- **DoD :** doc.
- **Effort :** 20 min.

### D.12 Align `config/training/sarl.yaml.num_frames` à Table 11
- **Faire :**
  - Question B.1 : 500k (table) ou 1M (texte) ? Trancher (défaut table appendix = 500k ; si doute, 1M comme plus conservateur).
  - Mettre à jour config.
  - Documenter choix dans `deviations.md`.
- **Vérifier :** config lit OK.
- **DoD :** commit.
- **Effort :** 20 min.

### D.13 Implémenter `_persist_checkpoint` + `_restore_from_checkpoint`
- **Préambule :** sans checkpoint/resume, settings 5-6 ne peuvent pas tourner (→ 4-9 jours wall vs 24h tier b3 max).
- **Faire :**
  - Nouvelle fonction `_persist_checkpoint(state, frame, out_dir)` : sauve policy_net, target_net, second_order_net, optimizer1, optimizer2, scheduler1, scheduler2, replay_buffer, random states Python/NumPy/Torch, total_frames, episode_returns. Format `.pt`.
  - `_restore_from_checkpoint(out_dir) -> state` : réciproque.
  - Fréquence : tous les `checkpoint_every_frames=50_000` (configurable). Rotate-keeps-last-3.
  - `run_training` accepte `resume: bool = False` → check checkpoint existant, reload si trouvé.
- **Vérifier :** unit test `tests/unit/experiments/sarl/test_checkpoint_resume.py` qui run 10k frames → checkpoint → destroy in-memory state → resume → run 10k more → compare total = 20k frames ciblé et return sequence cohérente.
- **DoD :** commit `feat(sarl): checkpoint + resume mid-run training (D-NNN)`.
- **Effort :** 4 h.

### D.14 Ajouter flag `--resume` à `scripts/run_sarl.py`
- **Faire :** CLI arg `--resume` → propage à `run_training(resume=True)`. Le `sarl_array.sh` existant (Sprint-07) passe déjà par le chemin idempotent ; ajouter `--resume` aussi pour forcer la reprise.
- **Vérifier :** smoke : submit job, cancel après 5k frames, resume → run complete OK.
- **DoD :** commit.
- **Effort :** 1 h.

### D.15 Update `tests/parity/sarl/_reference_sarl.py` — paper-faithful slices
- **Faire :** si décision C.9 était "paper wins", regénérer les slices de référence depuis `external/paper_reference/sarl_maps.py` avec les fix papier (alpha=0.45, lr_2=2e-4, etc.). Sinon garder tel quel et flagger qu'ils testent historical maps.py.
- **Vérifier :** `pytest tests/parity/sarl` vert.
- **DoD :** commit.
- **Effort :** 1-2 h selon décision.

### SARL+CL (4 fichiers)

### D.16 Review `src/maps/experiments/sarl_cl/loss_weighting.py`
- **Faire :** `DynamicLossWeighter` — running-max normalization. Comparer à paper §"Continual Learning" normalization "using the maximum individual loss observed throughout epochs".
- **Vérifier :** doc `docs/reviews/sarl_cl-loss_weighting.md`.
- **DoD :** doc.
- **Effort :** 1 h.

### D.17 Review `src/maps/experiments/sarl_cl/model.py` (257 L)
- **Faire :** `SarlCLQNetwork`, `SarlCLSecondOrderNetwork`. Variable channels handling (paper §CL "max input channels = 10 zero-padded 1×1 conv"). Comparer à `sarl_cl_maps.py`.
- **Vérifier :** doc.
- **DoD :** doc.
- **Effort :** 1.5 h.

### D.18 Review `src/maps/experiments/sarl_cl/trainer.py` (371 L)
- **Faire :** `sarl_cl_update_step`, 3-loss hybrid (L_reg + L_feature + L_task). Paper eq.15-18. Vérifier backward order (load-bearing).
- **Vérifier :** doc.
- **DoD :** doc.
- **Effort :** 1.5 h.

### D.19 Review `src/maps/experiments/sarl_cl/training_loop.py` (640 L)
- **Faire :** curriculum ordering (paper : Breakout → SpaceInvaders → Seaquest → Freeway, pas Asterix), 100k steps per env.
- **Vérifier :** doc.
- **DoD :** doc.
- **Effort :** 1.5 h.

### D.20 Fixes SARL+CL — align config weights + 100k steps
- **Faire :**
  - `config.cl.weight_task: 0.3`, `weight_distillation: 0.6`, `weight_feature: 0.1` (paper Table 11).
  - `config.training.num_frames: 100_000` (paper §CL).
  - Documenter divergences.
- **Vérifier :** smoke CL run OK.
- **DoD :** commit.
- **Effort :** 45 min.

### D.21 Checkpoint/resume SARL+CL
- **Faire :** adapter D.13 au CL (ajouter teacher_state, curriculum_stage_idx au checkpoint).
- **Vérifier :** test resume cross-env.
- **DoD :** commit.
- **Effort :** 2 h.

### D.22 Update `tests/parity/sarl_cl/` — paper-faithful
- **Faire :** idem D.15 pour CL.
- **Vérifier :** `pytest tests/parity/sarl_cl` vert.
- **DoD :** commit.
- **Effort :** 1 h.

### Blindsight (2 fichiers)

### D.23 Review `src/maps/experiments/blindsight/trainer.py` (442 L)
- **Faire :** `BlindsightTrainer.pre_train`, `evaluate`. Comparer à `blindsight_tmlr.py` et paper §2.2 Know-Thyself. RG-002 ouvert.
- **Vérifier :** doc `docs/reviews/blindsight-trainer.md`.
- **DoD :** doc + diagnostic RG-002.
- **Effort :** 2 h.

### D.24 Review `src/maps/experiments/blindsight/data.py`
- **Faire :** génération 400 patterns (paper §A.1), 3 conditions (suprathreshold, subthreshold, low vision).
- **Vérifier :** doc.
- **DoD :** doc.
- **Effort :** 45 min.

### D.25 RG-002 fix (Blindsight headline metric)
- **Faire :** selon diagnostic D.23, aligner la métrique d'eval avec celle du paper (test wager aggregated over conditions, pas discrimination single-condition). Peut nécessiter refactor `aggregate_perceptual.py`.
- **Vérifier :** 10 seeds smoke (local Mac) : new metric ≈ 0.97 (paper) au lieu de 0.755.
- **DoD :** RG-002 résolu ou escalade avec diagnostic honnête.
- **Effort :** 3 h.

### AGL (2 fichiers)

### D.26 Review `src/maps/experiments/agl/trainer.py` (405 L)
- **Faire :** `AGLTrainer.pre_train` + `evaluate`. RG-003 ouvert : downstream training phase post pre_train (après reset first-order) n'est pas portée. Paper page 13 §Results AGL le décrit.
- **Vérifier :** doc.
- **DoD :** doc + plan port downstream phase.
- **Effort :** 2 h.

### D.27 Review `src/maps/experiments/agl/data.py` (334 L)
- **Faire :** génération strings 3-8 letters, grammar A/B/random.
- **Vérifier :** doc.
- **DoD :** doc.
- **Effort :** 1 h.

### D.28 RG-003 fix — porter downstream training AGL
- **Faire :** porter la phase "supervised training on Grammar A vs B using the pre-trained 2nd-order" depuis `agl_tmlr.py`. C'est une nouvelle boucle + nouveau loss (BCE sur output vs grammar label). Intégrer dans `AGLTrainer.train()` (distinct de `pre_train`).
- **Vérifier :** 10 seeds smoke → classif ≈ 0.66 (high) / 0.62 (low) au lieu de ~chance.
- **DoD :** RG-003 résolu + test unit couvrant downstream training.
- **Effort :** 4-6 h.

### D.29 Update `tests/parity/` pour Blindsight + AGL si paper-faithful
- **Faire :** idem D.15/D.22.
- **Vérifier :** parity verte.
- **DoD :** commit.
- **Effort :** 1 h.

### D.30 Commit final Phase D
- **Faire :** série de commits déjà faits ; commit final `docs(reviews): per-experiment audit docs and fixes (Phase D)`.
- **Vérifier :** full pytest suite verte ; `experiment_matrix.md` à jour pour B+A+G (mais pas encore F).
- **DoD :** Phase D clôturée.
- **Effort :** 15 min.

---

## Phase E — Extensions (7-15 j)

### E.1 ACB — lire `external/paper_reference/sarl_ac_lambda.py` + paper §Results SARL ACB
- **Faire :** comprendre Actor-Critic avec trace decay λ=0.8 (Young & Tian 2019). Documenter mapping vers architecture modulaire.
- **Vérifier :** doc `docs/reviews/acb.md` + choix de namespace (`src/maps/experiments/sarl/acb.py` ou sous-dossier).
- **DoD :** doc.
- **Effort :** 2 h.

### E.2 Port ACB dans `src/maps/experiments/sarl/acb.py`
- **Faire :** nouveau fichier. Architecture : actor (policy head) + critic (value head) + trace decay λ. Pas de cascade, pas de second-order (c'est un baseline).
- **Vérifier :** forward shapes match paper ; actor outputs action distribution, critic outputs V(s).
- **DoD :** commit `feat(sarl): port ACB (Actor-Critic Baseline, setting 7) (E-NNN)`.
- **Effort :** 4 h.

### E.3 Étendre `setting_to_config` pour setting 7
- **Faire :** `training_loop.setting_to_config` supporte `setting=7` → ACB config (pas de meta, pas de cascade, utilise ACB network au lieu de DQN).
- **Vérifier :** `run_sarl.py --setting 7` dispatch vers ACB.
- **DoD :** commit.
- **Effort :** 1 h.

### E.4 Tests parité ACB
- **Faire :** `tests/parity/sarl/test_tier4_acb.py` : forward match `sarl_ac_lambda.py` référence à atol=1e-6.
- **Vérifier :** test vert.
- **DoD :** commit.
- **Effort :** 1.5 h.

### E.5 Smoke run ACB (1 seed, 1 jeu, 50k frames, CPU)
- **Faire :** smoke sbatch, vérifier `metrics.json` produit + rewards cohérentes.
- **Vérifier :** `metrics.json` existe + ordre de grandeur rewards.
- **DoD :** smoke log.
- **Effort :** 1 h.

### E.6 MeltingPot — Python 3.10 venv dédié sur $SCRATCH
- **Préambule :** dmlab2d wheel = Python 3.10 uniquement. Notre venv principal est 3.12 → co-existence.
- **Faire :** `uv venv --python 3.10 $SCRATCH/maps/venv-marl`. Documenter dans `docs/install_meltingpot.md`.
- **Vérifier :** `source $SCRATCH/maps/venv-marl/bin/activate && python --version` = 3.10.
- **DoD :** venv fonctionnel.
- **Effort :** 30 min.

### E.7 MeltingPot — installer dmlab2d wheel
- **Faire :**
  ```bash
  wget https://files.pythonhosted.org/packages/4b/31/884879224de4627b5d45b307cec8f4cd1e60db9aa61871e4aa2518c6584b/dmlab2d-1.0.0_dev.10-cp310-cp310-manylinux_2_31_x86_64.whl
  pip install dmlab2d-*.whl
  python -c "import dmlab2d; print(dmlab2d.__version__)"
  ```
- **Vérifier :** `import dmlab2d` sans erreur.
- **DoD :** dmlab2d installé.
- **Effort :** 30 min.

### E.8 MeltingPot — git clone + pin commit
- **Faire :**
  ```bash
  git clone https://github.com/deepmind/meltingpot /tmp/meltingpot
  cd /tmp/meltingpot && git log --oneline | awk '$1 < "2025-05-29"' | head -5
  git checkout <SHA>
  pip install --editable .[dev]
  ```
- **Vérifier :** `import meltingpot` OK + substrate list contient les 4 du papier.
- **DoD :** `docs/install_meltingpot.md` avec SHA pinné.
- **Effort :** 1 h.

### E.9 MeltingPot — smoke test 4 substrates
- **Faire :** script `scripts/smoke_meltingpot.py` : instancie `commons_harvest__closed`, `commons_harvest__partnership`, `chemistry__three_metabolic_cycles_with_plentiful_distractors`, `territory__inside_out`. Run 100 steps random policy. Vérifier shape observation 11×11 RGB (paper Table 2).
- **Vérifier :** 4 substrates instanciables + 100 steps sans crash.
- **DoD :** commit.
- **Effort :** 2 h.

### E.10 MARL — lecture complète `MARL/MAPPO-ATTENTIOAN/` + mapping functionnel
- **Faire :** doc `docs/reviews/marl-mapping.md` : mapping module-by-module des ~188 fichiers MAPPO-ATTENTIOAN vers la cible `src/maps/experiments/marl/` (6 fichiers cible).
- **Vérifier :** doc + diagramme structural.
- **DoD :** doc.
- **Effort :** 1 j.

### E.11 MARL — `src/maps/experiments/marl/model.py`
- **Faire :** Encoder conv + positional encoding (sinusoïdal relatif) + linear + GRU + second-order. Paper fig 4 + Table 12 (hidden_size=100).
- **Vérifier :** forward shapes match MAPPO-ATTENTIOAN sur une obs dummy.
- **DoD :** commit.
- **Effort :** 4 h.

### E.12 MARL — `src/maps/experiments/marl/buffer.py` (on-policy)
- **Faire :** PPO rollout buffer (pas replay buffer comme SARL).
- **Vérifier :** unit test shapes.
- **DoD :** commit.
- **Effort :** 2 h.

### E.13 MARL — `src/maps/experiments/marl/rollout.py`
- **Faire :** multi-agent env interaction, parallel envs (MAPPO pattern).
- **Vérifier :** smoke 100 steps sur 1 substrate.
- **DoD :** commit.
- **Effort :** 3 h.

### E.14 MARL — `src/maps/experiments/marl/losses.py`
- **Faire :** PPO clip loss + value loss + entropy + wagering BCE + cascade management.
- **Vérifier :** unit tests per-loss.
- **DoD :** commit.
- **Effort :** 3 h.

### E.15 MARL — `src/maps/experiments/marl/trainer.py`
- **Faire :** `marl_update_step` = MAPPO update + branche meta (2nd-order + cascade) — analogue à `sarl.trainer`.
- **Vérifier :** unit test forward + backward shapes.
- **DoD :** commit.
- **Effort :** 4 h.

### E.16 MARL — `src/maps/experiments/marl/training_loop.py`
- **Faire :** `run_training` + `setting_to_config` (7 settings pour MARL comme SARL) + _persist_outputs.
- **Vérifier :** smoke 1k steps 1 seed.
- **DoD :** commit.
- **Effort :** 3 h.

### E.17 MARL — `config/training/marl.yaml` aligné Table 12
- **Faire :** hidden=100, actor_lr=7e-5, weight_decay=1e-5, ppo_epoch=15, entropy=0.01, clip=0.2, num_env_steps (texte 300k, table 15e6 — trancher B.4).
- **Vérifier :** config parse OK.
- **DoD :** commit.
- **Effort :** 30 min.

### E.18 MARL — `scripts/run_marl.py`
- **Faire :** CLI typer, mêmes conventions que `run_sarl.py`.
- **Vérifier :** `uv run python scripts/run_marl.py --help`.
- **DoD :** commit.
- **Effort :** 1 h.

### E.19 MARL — `scripts/slurm/marl_array.sh`
- **Faire :** sbatch array 84 cellules (4 substrates × 7 settings × 3 seeds). `--gpus-per-node=h100:4` (MARL GPU obligatoire). `--time`, `--mem` conservateurs ; calibrer après E.21.
- **Vérifier :** `sbatch --test-only` OK.
- **DoD :** commit.
- **Effort :** 1 h.

### E.20 MARL — `tests/parity/marl/` (Tier 1/2/3)
- **Faire :** parity vs MAPPO-ATTENTIOAN : Tier 1 forward, Tier 2 buffer, Tier 3 update. `_reference_marl.py` extraits.
- **Vérifier :** atol=1e-6 sur chaque tier.
- **DoD :** commit.
- **Effort :** 4 h.

### E.21 MARL — smoke run 1 substrate × 1 seed × 10k steps (Tamia H100)
- **Faire :** sbatch smoke, attendre résultat, valider `metrics.json`.
- **Vérifier :** wall, fps, VRAM.
- **DoD :** smoke log + fps mesuré.
- **Effort :** 2 h + queue.

### E.22 Commit final Phase E
- **Faire :** `feat(extensions): ACB setting 7 + MeltingPot install + MARL port (Phase E)`.
- **DoD :** Phase E clôturée.
- **Effort :** 10 min.

---

## Phase F — Runs reproduction (2-4 semaines wall-clock)

**But :** produire les chiffres qui reproduisent tables 5-7 du papier. Discipline chainage obligatoire.

### F.1 Blindsight grid (local Mac ou CPU Tamia, 500 seeds × 7 settings = 3500 runs)
- **Faire :** grille complète. Sur Mac local (paper a fait ça sur RTX3070).
- **Vérifier :** 3500 × `metrics.json`.
- **DoD :** aggregate + rapport intermédiaire.
- **Effort :** ~12h wall (paper dit 12h pour 500 seeds / 1 setting).

### F.2 AGL grid (idem)
- Idem.

### F.3 SARL grid (Tamia, chaînée)
- **Faire :** 5 jeux × 7 settings × 3 seeds = 105 cellules. Chainées 2 par 2 via `submit_chained.sh`.
- **Vérifier :** 105 × `metrics.json` sous `$SCRATCH/maps/outputs/sarl/`.
- **DoD :** agrégation.
- **Effort :** 1-2 semaines wall.

### F.4 SARL+CL grid
- 4 envs × 7 settings × 3 seeds × 100k steps. Courtes.
- **Effort :** quelques jours wall.

### F.5 MARL grid
- 4 substrates × 7 settings × 3 seeds × 300k steps. GPU obligatoire.
- **Effort :** 1-2 semaines wall.

---

## Phase G — Rapport (2-3 j)

### G.1 Aggregate par domaine
- **Faire :** `scripts/aggregate_{perceptual,sarl,sarl_cl,marl}.py` tournent sur les outputs de F. Produisent `outputs/reports/*_summary.{json,md}`.
- **DoD :** 4 summaries.

### G.2 Table comparative papier vs reprod
- **Faire :** `docs/reports/paper-reproduction-report.md` : per-setting per-env, our mean ± std, z-score, verdict ±2σ.
- **DoD :** table finale.

### G.3 Déviations documentées
- **Faire :** finaliser `deviations.md` avec impact mesuré de chaque divergence.
- **DoD :** doc exhaustif.

### G.4 Update `experiment_matrix.md` avec valeurs mesurées finales
- **DoD :** matrice finale.

### G.5 PR repro/paper-faithful → main
- **Faire :** PR consolidée. Revue Rémy.
- **DoD :** mergé.

---

## Changelog

- 2026-04-19 (init) — plan créé en réponse au mandat Rémy "tout ce qui permet de refaire le papier doit rester + review composante par composante + intégrer MeltingPot/ACB/alignement hyperparams + respecter queue partagée + storage scratch". 82 sous-phases, séquentielles strict.
- 2026-04-19 (B.1 correction) — B.1 Vérifier: `24 hyperparams` → `23 hyperparams`. Recomptage depuis paper Table 11 p. 30 = 23 data rows (19 SARL + 4 CL). Contenu extrait conforme verbatim.
