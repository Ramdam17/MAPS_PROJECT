# Sprint 12 — Blindsight rewrite + tests + paper reproduction

**Status:** 🔵 open (2026-05-25)
**Branch:** `refactor/blindsight` (à brancher depuis `refactor/core`)
**Owner:** Rémy Ramadour
**Depends on:** Sprint 11 ✅ (core/) + Sprint 11.5 (cleanup) + Sprint 11.6 (glossaire + Makefile)
**Numerical reference:** `external/paper_reference/blindsight_tmlr.py` (verbatim
Vargas code — produit les chiffres paper Table 5a).

---

## Context

Sprint 11 a posé `core/` (cascade + second_order + losses, 67/67 tests).
Sprint 12 monte d'un cran : ajouter le **bloc Blindsight complet end-to-end**
en passant par les `utils/` et `networks/` qui le rendent possible.

Blindsight est le **premier domaine** que l'on porte — c'est le plus simple
(input binaire 100-dim, dataset synthétique, pas de RL). Si la chaîne tient
ici, elle tiendra sur AGL (Sprint 13). Les vraies montées en complexité
arrivent avec SARL (Sprint 14, MinAtar) puis MARL (Sprint 16, MeltingPot).

## Goal

Produire un Blindsight **paper-faithful end-to-end** :
- charge sa config depuis `config/domains/blindsight/{env,training}.yaml`
- entraîne avec `core/` (cascade + second_order + losses)
- évalue avec les métriques `testing()` du student
- reproduit les chiffres paper Table 5a (discrimination ~0.97, wager ~0.85
  pour le Full MAPS setting)
- supporte les 6 settings factoriels (paper Table 5a)
- supporte **les deux losses first-order** : CAE (default, student-parity)
  et SimCLR (D11.7, nouveau — empirical comparison post-Sprint 12)

**Pattern méthodologique nouveau (emprunté à bilevel-fishery)** : chaque
phase du sprint inclut une **table de reverse-prompts + corrections d'audit**
qui logge ce que le LLM a proposé et ce qu'on a refusé/modifié. Trace
explicite du dialogue de décision.

---

## Open items — décisions Day-1 (✅ tranchées 2026-05-25)

7 décisions identifiées en lisant les reverse-prompts utils + networks +
Blindsight. Toutes tranchées en Phase 12.A.

### D12.1 — Typed schema layer over OmegaConf ?

`utils/config.py` utilise OmegaConf pur (Hydra-light). Option d'ajouter
des dataclasses pydantic/`OmegaConf.structured` pour validation + autocomplete
IDE.

→ **Décidé** : **NON** (YAGNI). Garder OmegaConf pur. Si Sprint 14 SARL
+ Sprint 15 SARL+CL accumulent des typos config silencieux, on revisitera
post-Phase F.

### D12.2 — `utils/device.py` API ?

NEW module per Sprint 10 structure-decision. Auto-detect CPU/MPS/CUDA pour
le développement Mac local et HPC.

→ **Décidé** : `get_device(prefer: Literal["auto", "cpu", "mps", "cuda"] = "auto") → torch.device`
+ helper `pick_best_available() → torch.device`. ~30 LOC.

### D12.3 — `FirstOrderMLP.hidden_dim` Blindsight default ?

Trois choix tracés dans `docs/reproduction/deviations.md` :
- **40** (student `main()`) — current config, RG-002 H5
- **60** (paper Table 9 littéral) — *jamais* utilisé en pratique
- **100** (port Sprint 09 original) — no bottleneck

→ **Décidé** : **40** (current config, matche RG-002 résolution D.25).
Override disponible via `-o first_order.hidden_dim=60` pour reproduire
la lecture littérale du paper.

### D12.4 — SimCLR augmentation strategy pour stimulus binaires (D11.7 deferral)

LE choix de recherche du sprint. Le `simclr_loss` est porté Sprint 11.D
(NT-Xent math), mais générer la paire positive `z_j` depuis le stimulus
`z_i` est domain-specific et **non documenté** ni dans le paper ni dans
aucun port existant.

Options réalistes pour stimulus binaire 100-dim Blindsight :

| Option | Mécanisme | Param libre | Justification |
|--------|-----------|-------------|---------------|
| **A. Bit-flip aléatoire** | Flip `p%` des bits choisis uniformément | `p ∈ [0.05, 0.3]` | Le plus naturel pour binaire. Analogue noise-level perturbation. |
| **B. Bit-dropout** | Mask `p%` des bits à 0 | `p ∈ [0.1, 0.5]` | Asymétrique (zéros seulement). Plus brutal. |
| **C. Noise level resampling** | Re-tirer un sample même condition (SUPERTHRESHOLD) avec un nouveau RNG draw | — | Exploite le pipeline data existant. Pas "augmentation" stricto sensu. |
| **D. Combiné A+C** | Bit-flip ET re-tirer noise | `p` + condition | Plus de variance, possiblement trop. |

→ **Décidé** : **Option A — bit-flip p=0.1** (10% des bits flippés par
paire positive). C'est le baseline le plus simple et le plus faux-proof.
Sprint 12.D expose `p` en config (`simclr.augmentation.bit_flip_prob`)
pour sweep ultérieur. Si la comparaison empirique CAE vs SimCLR
(post-Sprint 12) montre que SimCLR sous-performe drastiquement, revisiter
les autres options.

### D12.5 — Trainer architecture : port direct vs refactor cleaner

Student `pre_train` + `testing` sont 522 lignes inline avec beaucoup de
state-passing. Options :

- **Port verbatim** : préserve parity 1-to-1, hard à lire, hard à étendre
- **Refactor `BlindsightTrainer` class** : `build()` + `train()` + `evaluate()`,
  state encapsulé, plus testable

→ **Décidé** : **refactor class** (suivre le reverse-prompt). Parity
bit-exact maintenue par les tests, pas par le 1-to-1 LOC.

### D12.6 — `BlindsightSetting` schema : 6-cell only ou backward-compat 2x2 ?

Le 2x2 schema (`cascade` boolean unique) est legacy Sprint 08. Le 6-cell
schema (`cascade_1st` + `cascade_2nd` séparés) est paper Table 5a et
résout le bug D.31 (Setting 4 ≠ Setting 6).

→ **Décidé** : **6-cell only**. Pas de backward-compat —
`refactor/blindsight` est une réécriture, pas un upgrade. Le 2x2 reste
sur `main` si besoin.

### D12.7 — Parity test scope : quels Tiers ?

- **Tier 1** = forward seul, bit-exact
- **Tier 2** = forward + backward, gradients bit-exact
- **Tier 3** = forward + backward + optimizer step, weights bit-exact
- **Tier 4 (proposé)** = `train(n_epochs=2)` end-to-end, losses bit-exact

→ **Décidé** : **Tier 1+2+3+4-light** (les 4 tiers — choix plus strict
que la reco). Tier 3 (1 update step isolé) ajoute un filet entre Tier 2
et Tier 4-light : si train diverge, Tier 3 isole le bug à l'étape
optimizer.step(). Coût marginal (1 test supplémentaire).

---

## Done when

### Code (target ~1200 LOC across all modules)

**Utils complement** :
- [ ] `src/maps/utils/config.py` (OmegaConf-backed YAML loader avec
      composition `defaults:`, dot-path overrides — D12.1)
- [ ] `src/maps/utils/logging_setup.py` (stdlib logging, format
      `%(asctime)s %(name)s [%(levelname)s] %(message)s`, optional file
      handler)
- [ ] `src/maps/utils/device.py` (auto-detect CPU/MPS/CUDA — D12.2)
- [ ] `src/maps/utils/paths.py` (load `config/paths.yaml`, expose
      `scratch_root` + `outputs_root` for HPC vs dev)
- [ ] `src/maps/utils/__init__.py` exporte les symboles publics

**Networks** :
- [ ] `src/maps/networks/first_order_mlp.py` (`FirstOrderMLP` encoder-decoder
      MLP, no bias, weight init uniform(-1, 1) — D12.3 hidden_dim default)
- [ ] `src/maps/networks/__init__.py`

**Blindsight domain** :
- [ ] `src/maps/domains/blindsight/data.py` (`generate_patterns()`,
      `StimulusCondition` enum, `TrainingBatch` dataclass)
- [ ] `src/maps/domains/blindsight/augmentations.py` (`bit_flip(x, p)`
      pour SimCLR positive pairs — D12.4)
- [ ] `src/maps/domains/blindsight/trainer.py` (`BlindsightTrainer`
      class — D12.5 refactor, supporte `first_order_loss.kind ∈ {cae, simclr}`)
- [ ] `src/maps/domains/blindsight/cli.py` (Typer CLI, single-run + all-settings
      mode)
- [ ] `src/maps/domains/blindsight/__init__.py`

### Tests (target ~80 tests)

**Unit tests** :
- [ ] `tests/unit/utils/test_config.py` : load_config, composition,
      overrides, errors
- [ ] `tests/unit/utils/test_logging_setup.py` : level resolution, file
      handler, idempotency
- [ ] `tests/unit/utils/test_device.py` : auto-detect, prefer override,
      fallback
- [ ] `tests/unit/utils/test_paths.py` : scratch_root resolution, env
      override
- [ ] `tests/unit/networks/test_first_order_mlp.py` : forward shape,
      weight init range, no-bias check, chunked vs global sigmoid,
      cascade asymmetry (h1 no-op vs h2)
- [ ] `tests/unit/domains/blindsight/test_data.py` : shape contract,
      stimulus distribution check, threshold semantics
- [ ] `tests/unit/domains/blindsight/test_augmentations.py` : bit_flip
      changes p% of bits, idempotent on p=0, max-noise on p=1
- [ ] `tests/unit/domains/blindsight/test_trainer.py` : build → train(2 epochs)
      → evaluate smoke ; setting dispatch (cascade_1st, cascade_2nd,
      second_order flags)

**Parity tests** :
- [ ] `tests/parity/blindsight/test_data_parity.py` (Tier 1) : bit-exact
      `generate_patterns()` vs student verbatim extract
- [ ] `tests/parity/blindsight/test_first_order_mlp_parity.py` (Tier 1+2) :
      forward + backward bit-exact vs student
- [ ] `tests/parity/blindsight/test_update_step_parity.py` (Tier 3 —
      D12.7 user choice) : 1 isolated `forward + backward + optimizer.step()`
      bit-exact ; weights post-update match student to 1e-5
- [ ] `tests/parity/blindsight/test_trainer_parity.py` (Tier 4-light) :
      `BlindsightTrainer.train(n_epochs=2, seed=42)` losses match student
      to 1e-4 (accumule plus de drift sur le seq de losses)

**Integration smoke** :
- [ ] `tests/integration/blindsight/test_smoke.py` : 1 run end-to-end
      Setting 6 Full MAPS, seed=42, n_epochs=10 → loss decreases,
      eval metrics non-trivial

### Reproduction (informational, post-Phase F-equivalent)

- [ ] **Goal benchmark** : reproduction du paper Table 5a Setting 6
      (discrimination ~0.97, wager ~0.82–0.85). Run via `make test-slow`
      Sprint 12.G.
- [ ] Si gap, log dans `docs/reproduction/deviations.md` avec hypothèses.

### Doc

- [ ] Reverse-prompts mis à jour pour les chemins ports (déjà fait Sprint
      11.5, vérifier)
- [ ] `docs/learning/walkthroughs/blindsight-from-paper-to-code.md` —
      court (3-4 pages) : pourquoi le `multiplier/2` threshold crée
      l'effet blindsight, pourquoi le two-loss gradient pattern
      (loss_2.backward → optim_2.step → loss_1.backward → optim_1.step)
      est load-bearing
- [ ] `docs/learning/glossaire.md` — section "Ajouts du Sprint 12" en bas
      (bit-flip aug, two-loss gradient pattern, etc.)
- [ ] `docs/sprints/sprint-12-blindsight.md` — closeout avec commit hashes
- [ ] `CLAUDE.md` Current Status → Sprint 13 next

---

## Phases (sequencing recommandé)

| Phase | Effort | Tâches |
|-------|--------|--------|
| **12.A** — décisions D12.1→D12.7 | 1 session courte | Choisir, acter inline dans Open items |
| **12.B** — `utils/{config,logging_setup,device,paths}.py` + tests | 1-2 sessions | 4 modules, tous petits-moyens. Indépendants. |
| **12.C** — `networks/first_order_mlp.py` + tests | 1 session | Encoder/decoder MLP avec chunked-sigmoid factory. |
| **12.D** — `domains/blindsight/data.py` + `augmentations.py` + parity tests | 1-2 sessions | `generate_patterns()` Tier 1 parity (bit-exact). `bit_flip()` simple. |
| **12.E** — `domains/blindsight/trainer.py` + unit tests | 2-3 sessions | Le plus gros. Refactor class, two-loss pattern, eval métriques. |
| **12.F** — `domains/blindsight/cli.py` + smoke | 1 session | Typer CLI, save artifacts, output layout. |
| **12.G** — parity tests vs paper_reference end-to-end | 1-2 sessions | Tier 1+2+4-light. Si OOM ou divergence, investiguer. |
| **12.H** — walkthrough + closeout | 1 session | Walkthrough Blindsight + glossaire ajouts + sprint closeout |

---

## Pattern méthodologique nouveau — Reverse-prompts + corrections d'audit

Inspiré du bilevel-fishery `docs/bricks/00_skeleton.md`. Chaque phase
documentée inline dans ce spec avec table :

```markdown
### Phase 12.X — <titre>

**Reverse-prompts utilisés (et corrections d'audit)** :

| # | Source | Correction d'audit |
|---|--------|-------------------|
| 12.X.1 | docs/learning/reverse-prompts/path/to.md | (aucune / précision / refus) |
| 12.X.2 | ... | ... |
```

Pendant l'exécution du sprint, chaque commit met à jour la table
correspondante avec les décisions effectivement prises. Trace
historique du *dialogue* avec le LLM, pas juste de la décision finale.

---

## Tests strategy

**4 niveaux de garantie** (vs 3 en Sprint 11 — on ajoute Tier 4-light) :

1. **Math sanity (unit, toy tensors)** : prouve les contracts. Pas de
   dépendance externe. `tests/unit/{utils,networks,domains/blindsight}/`.

2. **Parity vs paper_reference Tier 1** (forward seul) : bit-exact pour
   `data.py`, `first_order_mlp.py`. Tolérance 1e-6.

3. **Parity vs paper_reference Tier 2** (forward + backward) : gradients
   bit-exact. Tolérance 1e-5 (50-iter cascade).

4. **Parity vs paper_reference Tier 4-light** (`train(n_epochs=2)` end-to-end) :
   sequence de losses bit-exact vs student. Tolérance 1e-4 (accumule
   plus de drift). C'est CE test qui certifie que le port reproduit le
   paper.

5. **Integration smoke** (`train(n_epochs=10)` Setting 6 Full MAPS) :
   loss decreases, eval metrics non-trivial. Pas paper-faithful, juste
   "ça tourne".

6. **Reproduction (Phase 12.G+, slow)** : Setting 6 Full MAPS sur 4-6
   seeds × default `n_epochs=200` (paper convention). Compare aux z-scores
   Table 5a. Goal : discrim ≥ 0.94, wager ≥ 0.80.

---

## Garde-fous

- **Aucune modification de `core/`** sans bug urgent. Sprint 11 a verrouillé
  les API. Si Sprint 12 demande une nouvelle API core (e.g.
  `cascade_update` qui prend un masque), ouvrir une issue/discussion
  d'abord.
- **Aucun changement de hyperparams paper** sans entrée dans `deviations.md`.
  Tout vient de `config/maps.yaml` + `config/domains/blindsight/`, jamais en
  hardcoded.
- **Aucun `print()`** : `logging` partout.
- **Type hints obligatoires** + `from __future__ import annotations`.
- **Tests parity Sprint 11 doivent rester verts** (67/67) pendant tout Sprint 12.

---

## Hors scope (explicitement)

- AGL (Sprint 13) — partage `networks/first_order_mlp.py` mais a sa
  propre `data.py`, `pool.py`, `trainer.py`, `cli.py`.
- SARL (Sprint 14), SARL+CL (Sprint 15), MARL (Sprint 16), METTA (Sprint 16+).
- Comparison empirique CAE vs SimCLR : livrable post-Sprint 12
  (`docs/reports/cae-vs-simclr-blindsight.md`).
- Unification `core.cae_loss` vs `domains/sarl/losses.cae_loss` (DETTE-2) :
  Sprint 14 ou post-Phase F.
- Energy tracker (DETTE-4) : Sprint 16 (MARL).
- Runs Compute Canada (Narval) : post-Sprint 16.

---

## Risques identifiés

- **R1 — Tier 4-light parity peut échouer** pour des raisons subtiles
  (ordre RNG calls : data RNG vs network RNG vs dropout RNG). Si ça
  échoue, débugger via comparison étape-par-étape (générer 1 batch,
  comparer ; 1 forward, comparer ; 1 backward, etc.).

- **R2 — SimCLR augmentation choice (D12.4 = bit-flip 0.1)** peut donner
  des résultats catastrophiques (loss explose, no convergence). Mitigation :
  test sanity en Phase 12.D `train(n_epochs=10)` avec `first_order_loss.kind=simclr`,
  comparer le profil de loss à CAE.

- **R3 — RG-002 H5 (hidden_dim 40 vs 60) reste à vérifier** : la config
  est sur 40, mais la réelle reproduction des chiffres Table 5a doit
  être confirmée Sprint 12.G. Si gap persiste, hypothèses possibles :
  hidden_dim n'est pas le seul knob, autre interaction non identifiée.

- **R4 — `BlindsightTrainer` class refactor** peut dériver de la parity
  student. Mitigation : Tier 4-light test = filet de sécurité. Si ça
  diverge, on conserve `train()` mais on revisite la décomposition
  interne.

---

## Next sprint (Sprint 13 — préparation)

Sprint 12 livre `domains/blindsight/` end-to-end + `networks/first_order_mlp.py`
(réutilisable AGL) + `utils/` complet (config, logging, device, paths).

**Sprint 13 = AGL** :
- `domains/agl/{data.py, pool.py, trainer.py, cli.py}`
- Réutilise `networks/first_order_mlp.py` avec `make_chunked_sigmoid(6)`
- Réutilise tout `utils/`
- Référence num : `external/paper_reference/agl_tmlr.py`
- Reproduction Table 5b (paper)
- Risque clé : le `reset` first-order après pre-train (D-agl-reset) — le
  mécanisme de dissociation conscious/unconscious. Code-fragile.

Si Sprint 12 va bien, Sprint 13 devrait prendre ~50-70% du temps de
Sprint 12 (réutilisation forte).
