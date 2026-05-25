# Sprint 11 — MAPS core rewrite + tests

**Status:** ✅ done (2026-05-25)
**Branch:** `refactor/core` (branchée depuis `refactor/main-rewrite`)
**Owner:** Rémy Ramadour
**Effort réel :** 1 session intensive (Phase A → G en continu)
**Depends on:** Sprint 10 closed — structure + reverse-prompts validés
**Numerical reference:** `external/paper_reference/blindsight_tmlr.py` (verbatim
Vargas code — c'est CE code qui a produit les chiffres paper Tables 5/6/7,
pas la version Sprint 09 sur `main`).

## Closeout (2026-05-25)

**Phase 11.A ✅ — décisions Day-1 (commit `c498bb3`)** :
7 décisions tranchées (D11.1 → D11.7). Voir section Open items.

**Phase 11.B ✅ — utils/seeding + scaffold (commit `42ae8ba`)** :
- `src/maps/utils/seeding.py` : `set_all_seeds(seed)` (random, numpy,
  torch CPU/CUDA/MPS, PYTHONHASHSEED) + `LAB_DEFAULT_SEED = 42`
- `tests/conftest.py` : autouse `seed_everything` fixture
- `tests/{unit,parity}/core/__init__.py` : scaffold
- `tests/unit/utils/test_seeding.py` : 9/9 pass

**Phase 11.C ✅ — core/cascade.py (commit `f90dc0c`)** :
- `cascade_update(new, prev, cascade_rate)` (D11.1 renamed)
- `n_iterations_from_alpha(rate)`
- Docstrings citent McClelland1989, Vargas2025, D-sarl-cascade-noop,
  Gal & Ghahramani 2016
- 15/15 unit tests pass (incl. analytical no-op convergence)

**Phase 11.D ✅ — core/losses.py (commit `6c55665`)** :
- `cae_loss(weight, x, recons_x, hidden, lam, *, recon='bce_sum')`
  avec quirk `h(1-h)` sur ReLU preserved byte-for-byte
- `wagering_bce_loss(wager, target, *, reduction='mean')` — `pos_weight`
  retiré (D11.6)
- `weight_regularization(student, teacher)` — `zip(strict=True)`
- `simclr_loss(z_i, z_j, *, temperature=0.5, reduction='mean')` —
  NT-Xent réelle (D11.7), pas un stub
- `distillation_loss` **DELETED** (D11.5) — DETTE-3 résolue
- 21/21 unit tests pass (incl. CAE parity vs paper_reference verbatim,
  SimCLR cross-check vs manual cross_entropy)

**Phase 11.E ✅ — core/second_order.py (commit `24e89c2`)** :
- `ComparatorMatrix(nn.Module)` — eq.1 stateless avec shape validation
  (D11.4)
- `WageringHead(input_dim, *, n_wager_units=1, hidden_dim: int | None = None)`
  — D11.2 None default ; Pasquali hidden D.25 restored ; readout init
  uniform(0, 0.1)
- `SecondOrderNetwork(input_dim, *, n_wager_units, hidden_dim, dropout=0.5)`
  — attribut `self.wager` (D11.3) ; composition Comparator → Dropout
  → cascade → Wager ; caller threads `prev_comparison`
- 17/17 unit tests pass (incl. Pasquali hidden differentiation,
  eval-vs-train cascade divergence proving MC-dropout mechanism)

**Phase 11.F ✅ — parity tests vs paper_reference (commit `8d6d926`)** :
- `tests/parity/core/test_against_paper_reference.py` avec
  `_StudentSecondOrderNetwork` extrait verbatim de
  `external/paper_reference/blindsight_tmlr.py:213-252`
- 5/5 parity tests pass : init weights bit-identical, single-forward
  eval bit-exact (1e-7), 50-cascade eval (1e-6), 50-cascade train avec
  RNG-controlled dropout (1e-5), backward gradients (1e-5)

**Phase 11.G ✅ — walkthrough + closeout (ce commit)** :
- `docs/learning/walkthroughs/cascade-from-paper-to-code.md` —
  narratif post-rewrite (9 sections, expose équivalence MC-dropout
  et co-design de `core/`)
- `docs/reproduction/deviations.md` — refs `components/` → `core/`
  (D-blindsight-wager-hidden, D-agl-wager-hidden, DETTE-1, DETTE-2 ;
  DETTE-3 ✅ RESOLVED via D11.5)
- `docs/reproduction/paper_equations_extracted.md` + `paper_vs_code_audit.md`
  — refs path mis à jour, mention SimCLR D11.7
- `CLAUDE.md` Current Status → Sprint 12 next

### Suite de tests Sprint 11 — 67 passing

| Tier | Path | Count | Tolerance |
|------|------|-------|-----------|
| unit | `tests/unit/utils/test_seeding.py` | 9 | bit-exact |
| unit | `tests/unit/core/test_cascade.py` | 15 | bit-exact |
| unit | `tests/unit/core/test_losses.py` | 21 | 1e-6 (CAE parity), 1e-7 (sanity) |
| unit | `tests/unit/core/test_second_order.py` | 17 | 1e-5 (50-iter convergence) |
| parity | `tests/parity/core/test_against_paper_reference.py` | 5 | 1e-7 (eval), 1e-5 (train) |

Tous les tests parity sont des **gardiens cross-sprint** : tout
domaine qui les casse au Sprint 12+ a un bug de domaine, pas un test
à modifier.

### Insights scientifiques à reporter Sprint 12+

1. **`core/` est un bloc co-conçu.** Les 3 modules ne sont pas
   indépendants — le walkthrough cascade-from-paper-to-code l'explique
   en détail. Toute modification de `cascade.py` doit s'évaluer en
   tenant compte de `second_order.py` (qui le consomme) et
   `losses.py` (qui reçoit le gradient à travers).

2. **`simclr_loss` est porté mais non utilisé.** D11.7 a livré la
   math NT-Xent réelle. Sprint 12 Blindsight devra (a) écrire
   `domains/blindsight/augmentations.py` (choix de recherche : bit-flip
   p% ? noise level perturbation ?), (b) brancher `first_order_loss.kind
   ∈ {cae, simclr}` dans le trainer. La comparaison empirique CAE vs
   SimCLR est un livrable post-Sprint 12.

3. **`distillation_loss` est DELETED.** Sprint 15 SARL+CL ne pourra
   pas l'importer. Le vrai "distillation" CL est `weight_regularization`
   (L2 anchor) — pas un changement, c'est ce que le student utilisait
   déjà en prod.

4. **MC-dropout equivalence** (Gal & Ghahramani 2016) est *le* angle
   pédagogique central de MAPS. Il devrait apparaître dans tous les
   futurs walkthroughs et le README final.

### Hors scope reporté

- `networks/first_order_mlp.py` → Sprint 12 Blindsight
- `utils/{config, logging_setup, device, energy_tracker}.py` → Sprint 12
- Tous les domaines → Sprints 12-16
- Unification `core.cae_loss` vs `domains/sarl/losses.cae_loss` (DETTE-2)
  → Sprint 14 SARL ou post-Phase F
- Unification `SecondOrderNetwork` vs `SarlSecondOrderNetwork` (DETTE-1)
  → Sprint 14 ou post-Phase F
- SimCLR augmentations + comparaison empirique → Sprint 12+ et au-delà

---

## Context

Sprint 10 a démoli `src/maps/` et écrit 41 reverse-prompts. Sprint 11
ouvre la réécriture par le bloc le plus stable et le plus partagé : les
3 modules `core/`. Ils sont consommés par tous les domaines (Blindsight,
AGL, SARL, SARL+CL) et ne dépendent que de PyTorch + `config/maps.yaml`.

Les 3 modules forment un bloc co-conçu :

```
   ┌─────────────────────────┐
   │  cascade.cascade_update │  ← 6 sites d'appel cross-domain
   └───────────┬─────────────┘
               │
   ┌───────────┴─────────────┐
   │  second_order           │  ← effet réel SEULEMENT si dropout actif
   │  Comparator+Cascade+    │     (cf. cascade.md D-sarl-cascade-noop)
   │  Wager+Hidden Pasquali  │
   └───────────┬─────────────┘
               │ retain_graph=True
               │ x 50 unrolls
   ┌───────────┴─────────────┐
   │  losses.cae_loss        │  ← quirk h(1-h) sur ReLU universel
   │  losses.wagering_bce    │  ← équivalent BCE-with-logits
   └─────────────────────────┘
```

Réécrire les 3 ensemble (et leurs tests) évite les coupures sémantiques :
on ne peut pas changer `cascade_update` sans considérer comment
`SecondOrderNetwork` consomme `prev_comparison`, et on ne peut pas
changer `SecondOrderNetwork` sans considérer comment `cae_loss` reçoit
les gradients à travers le unroll de 50 itérations.

## Goal

Produire `src/maps/core/{cascade.py, second_order.py, losses.py}` —
paper-faithful, type-annotés, testés contre `external/paper_reference/`
— avec une suite de tests qui peut servir de **gardienne** pour tous
les domaines Sprint 12+.

**Pas de domaine** dans ce sprint. Pas de `FirstOrderMLP` (vit dans
`networks/` — Sprint 12 Blindsight). Le seul module hors-`core/` admis
est `utils/seeding.py` (10 lignes, nécessaire pour déterminisme des
tests).

---

## Open items — décisions Day 1 (✅ tranchées 2026-05-25)

Sept décisions identifiées en lisant les reverse-prompts core. Toutes
tranchées en Phase 11.A.

### D11.1 — `cascade_update` : `alpha` ou `cascade_rate` ?

Review C1-fix-3 recommande `cascade_rate` (clarté + distingue de l'autre
α du paper, EMA wagering=0.45). Signature actuelle student : `alpha`.

→ **Décidé** : `cascade_rate` (sémantique > tradition).

### D11.2 — `SecondOrderNetwork.hidden_dim` : `int = 0` ou `int | None = None` ?

`hidden_dim=0` est cryptique (= "pas de hidden"). `hidden_dim=None`
signale explicitement l'absence. Mais `int = 0` matche student.

→ **Décidé** : `int | None = None` (clarté API > parity visuelle).

### D11.3 — `wagering_head` vs `wager` (naming attribut) ?

Student inline : `self.wager`. Port Sprint 09 : `self.wagering_head`.
Koch & Preuschoff (2007) parlent de "betting" — `wager` plus proche du
vocabulaire littéraire.

→ **Décidé** : `wager` (vocabulaire littéraire Koch & Preuschoff).

### D11.4 — `ComparatorMatrix` : `nn.Module` ou fonction libre ?

Stateless (aucun paramètre). Garder `nn.Module` permet l'extension
hypothétique (comparator appris) et la cohérence `state_dict()`. Fonction
libre = -20 lignes, plus simple.

→ **Décidé** : `nn.Module` (extensibilité a un coût marginal, le retrait
est facile plus tard ; l'inverse ne l'est pas).

### D11.5 — `distillation_loss` : keep, demote, ou delete ?

**0 callers en prod** (DETTE-3, dead code). Le vrai "distillation" CL
est `weight_regularization` (L2 anchor EWC-style). Garder = pollue API.
Supprimer = perd fidélité 1-to-1 student.

→ **Décidé** : **delete**. YAGNI strict. Si Sprint 15 SARL+CL veut un
jour tester KL distillation à la place de L2 anchor, on récrira depuis
git history (`git show main:src/maps/components/losses.py`). Le `core/`
ne porte que ce qui tourne.

### D11.6 — `wagering_bce_loss.pos_weight` : rename, keep, ou remove ?

Bug API latent (paramètre mal nommé, 0 callers actuels). YAGNI direct.

→ **Décidé** : **remove**. Easy to re-add si besoin (1 ligne
`F.binary_cross_entropy(weight=...)`).

### D11.7 — SimCLR : stub ou implémentation réelle ?

Le YAML `config/maps.yaml` a déjà `first_order_loss.kind: cae | simclr`
(D-002 D.22b). Choix initial = stub `NotImplementedError`. Discussion a
révisé en sprint planning : Rémy veut **comparer empiriquement CAE vs
SimCLR** sur Blindsight Sprint 12 — donc nécessité d'une vraie loss.

→ **Décidé** : implémenter la **math NT-Xent en Sprint 11**
(`simclr_loss(z_i, z_j, *, temperature)` — ~30 lignes, pure PyTorch,
testable en isolation). **Augmentations** stimulus-binaires (bit-flip
p%, etc.) **reportées Sprint 12** (domain-specific, choix de recherche).
**Comparaison empirique** CAE vs SimCLR sur Blindsight reportée
post-Sprint 12.

Tests Sprint 11 = sanity (loss↓ quand `z_i ≈ z_j`, loss↑ orthogonal,
gradient flow, reduction modes). **Pas de parity** : aucune référence
externe pour SimCLR sur stimulus binaires.

Voir `docs/reproduction/deviations.md` D-002 et
`docs/reports/sprint-08-d22b-simclr-decision.md` pour le contexte
historique de cette divergence paper↔code.

---

## Done when

### Code

- [ ] `src/maps/core/cascade.py` écrit, type-annoté, NumPy-docstrings,
      citations Vargas2025 + McClelland1989 inline
- [ ] `src/maps/core/second_order.py` écrit (3 classes : `ComparatorMatrix`,
      `WageringHead`, `SecondOrderNetwork`), avec `hidden_dim` paramétrable
      (D.25 D.28 Pasquali restored)
- [ ] `src/maps/core/losses.py` écrit (4 fonctions : `cae_loss`,
      `wagering_bce_loss`, `weight_regularization`, `simclr_loss`).
      `distillation_loss` supprimée (D11.5). `pos_weight` param retiré
      de `wagering_bce_loss` (D11.6). `simclr_loss` implémentée réellement
      (math NT-Xent, pas un stub — D11.7).
- [ ] `src/maps/utils/seeding.py` écrit (`set_all_seeds(seed: int)` →
      random, numpy, torch, torch.cuda, torch.mps)
- [ ] `src/maps/core/__init__.py` exporte les symboles publics

### Tests (gardes-fous structurels pour tous les sprints suivants)

- [ ] `tests/unit/core/test_cascade.py` :
  - test analytique no-op : `cascade_update` appliquée N fois sur
    tensor déterministe identique ≡ 1 application (preuve fermeture)
  - test bounds : `α=0` raise, `α=-0.1` raise, `α=1.1` raise, `α=1.0`
    OK, `α=0.02` OK
  - test bootstrap : `prev=None` retourne `new` directement
  - test `n_iterations_from_alpha(0.02) == 50`
- [ ] `tests/unit/core/test_second_order.py` :
  - test `ComparatorMatrix(x, y) ≡ x - y` (eq.1, shapes variées)
  - test shape validation raise (broadcasting silencieux PyTorch évité)
  - test `WageringHead` output ranges :
    - `n_wager_units=1` → sigmoid ∈ [0, 1]
    - `n_wager_units=2` → raw logits (peut être négatif, PAS de softmax)
  - test `hidden_dim` injecté : `hidden_dim=10` ajoute
    `Linear(input, 10) → ReLU → Linear(10, n_wager_units)`
  - test threading `prev_comparison` : forward(t+1) avec output(t) ≠
    forward(t+1) avec `None` (cascade fait quelque chose)
- [ ] `tests/unit/core/test_losses.py` :
  - test `cae_loss` parity vs `external/paper_reference/blindsight_tmlr.py`
    (extraire la fonction CAE inline, comparer à 1e-6)
  - test quirk `h(1-h)` sur ReLU output : preserved byte-for-byte (pas
    de "correction" silencieuse)
  - test `wagering_bce_loss` ≡ `nn.BCELoss(reduction='sum')` pour les
    valeurs hors-bornes (numerical safety clamp_min(1e-12))
  - test `weight_regularization` :
    - identique → 0
    - 1 param diffère → contribution exacte
    - teacher non-frozen → vérifier qu'on a docstring qui prévient
  - test `simclr_loss` (NT-Xent — D11.7) :
    - **monotonie similarity** : loss(z_i, z_i+ε) < loss(z_i, z_random)
      pour ε petit (paire positive vs orthogonale)
    - **monotonie batch** : loss diminue quand on duplique des pairs
      positives dans le batch (signal qui domine)
    - **temperature scaling** : `temperature=1.0` vs `temperature=0.1`
      change la magnitude prédictiblement (haute τ = soft, basse τ =
      sharp)
    - **gradient flow** : `loss.backward()` met du gradient non-zero
      sur z_i et z_j
    - **reduction modes** : `mean` vs `sum` cohérents
    - **shape validation** : `z_i.shape != z_j.shape` raise
- [ ] `tests/parity/core/test_against_paper_reference.py` :
  - 1 forward+backward complet de `SecondOrderNetwork` avec input
    déterministe (seed fixé), comparé à la séquence
    `Comparator → Dropout → cascade (×50) → Wager` extraite
    `blindsight_tmlr.py` lines 180-225 (à identifier)
  - Tolérance 1e-5 (float32, 50 unrolls accumulent du bruit numérique)

### Doc

- [ ] `src/maps/core/cascade.py` docstrings citent McClelland (1989) +
      Vargas (2025) §2.1 eq.6 + l'ambiguïté D-sarl-cascade-noop dans
      les Notes
- [ ] `src/maps/core/second_order.py` docstrings citent Pasquali &
      Cleeremans (2010) + Koch & Preuschoff (2007) + D.25 Pasquali hidden
- [ ] `src/maps/core/losses.py` docstrings citent Rifai (2011) + Hinton
      (2015) + le quirk `h(1-h)` sur ReLU explicitement (C7-fix-2)
- [ ] `docs/learning/walkthroughs/cascade-from-paper-to-code.md` (court,
      3-4 pages) — narratif post-rewrite : pourquoi la cascade + dropout
      = MC-dropout, et pourquoi `core/` est conçu comme un bloc

### Reproduction

- [ ] `docs/reproduction/deviations.md` : mettre à jour les références
      modules (de `components/X.py` vers `core/X.py`). Pas de nouvelle
      déviation introduite.
- [ ] Garder D-002, D-003, D-001, D.25 explicitement liés depuis les
      docstrings.

---

## Phases (sequencing recommandé)

| Phase | Effort | Tâches |
|-------|--------|--------|
| **11.A** — décisions D11.1→D11.7 | 1 session courte | Choisir, acter dans le sprint doc en post-it `decided:` |
| **11.B** — `utils/seeding.py` + scaffold tests | 1h | Petit module, `conftest.py` global `seed_everything` fixture |
| **11.C** — `cascade.py` + ses tests | 1 session | Le plus simple, sert de gabarit |
| **11.D** — `losses.py` + ses tests | 1-2 sessions | Plus volumineux : `cae_loss` (parity paper_reference), `wagering_bce_loss` (sans `pos_weight`), `weight_regularization`, et `simclr_loss` NT-Xent (D11.7). `distillation_loss` deleted. |
| **11.E** — `second_order.py` + ses tests | 1-2 sessions | Dépend de cascade + utilise les losses dans les tests parity |
| **11.F** — `parity/test_against_paper_reference.py` | 1 session | Test d'intégration : un forward+backward end-to-end de `SecondOrderNetwork` vs paper_reference inline |
| **11.G** — walkthrough + closeout | 1 session | `cascade-from-paper-to-code.md` + sprint closeout |

---

## Tests strategy

**Trois niveaux de garantie** :

1. **Math sanity (unit, toy tensors)** : prouve que la formule est bien
   ce qui est écrit. Pas de dépendance externe. Vivent dans
   `tests/unit/core/`.

2. **Parity vs paper_reference (parity, extracted functions)** : compare
   bit-for-bit notre implémentation factorisée contre les implémentations
   inline du code Vargas original. C'est CE code qui a produit les
   chiffres paper — si on diverge ici, on diverge du paper. Tolérance
   1e-6 quand pure-math, 1e-5 quand 50 unrolls float32. Vivent dans
   `tests/parity/core/`.

3. **Intégration minimale** : un forward+backward complet de
   `SecondOrderNetwork` sur un input déterministe seedé, comparé à la
   séquence inline du paper_reference. Détecte les régressions
   structurelles (ordre dropout/cascade/wager inversé, etc.).

**Pas de test sur les domaines** (Blindsight, AGL, SARL) dans ce sprint
— ça vient avec les domaines. Mais les tests core écrits ici doivent
être **réutilisables tels quels** par les sprints suivants : un domaine
qui casse un test core, c'est un domaine qui casse, pas un test à
modifier.

---

## Garde-fous

- **Aucun domaine touché** dans ce sprint. Si on a envie d'écrire
  `domains/blindsight/data.py`, on stoppe et on l'ajoute au Sprint 12.
- **Aucune optimisation perf non testée** : le skip-cascade-déterministe
  (Q3 cascade.md) est tentant mais hors scope. Logger un TODO si on en
  a envie.
- **Aucun changement de hyperparam paper** sans entrée dans
  `deviations.md`. `α=0.02`, `dropout=0.5`, `λ_CAE=1e-4`, etc. : tout
  vient de `config/maps.yaml`, jamais en hardcoded.
- **`print()` interdit** : `logging` partout (sauf possiblement un CLI
  futur).
- **Type hints obligatoires** sur toutes les fonctions publiques.
  `from __future__ import annotations` en tête de chaque module.

---

## Hors scope (explicitement)

- Réécriture des domaines : Sprints 12-16.
- `networks/first_order_mlp.py` : Sprint 12 (Blindsight, qui le partage
  avec AGL Sprint 13).
- Reste de `utils/` (config, logging_setup, device, energy_tracker) :
  Sprint 12 (Blindsight les nécessite tous).
- Unification `cae_loss` core vs sarl/losses (DETTE-2) : Sprint 14 SARL
  ou post-Phase F.
- Unification `SecondOrderNetwork` vs `SarlSecondOrderNetwork` (DETTE-1)
  : Sprint 14 SARL ou post-Phase F.
- Optimisation perf cascade skip : à valider analytiquement d'abord,
  pas dans ce sprint.
- SimCLR **augmentations stimulus-binaires** (bit-flip, dropout
  d'éléments, etc.) : Sprint 12 Blindsight. Le `simclr_loss` math est
  porté Sprint 11 (D11.7) mais le pipeline d'augmentation positives
  est domain-specific.
- SimCLR **comparaison empirique CAE vs SimCLR** sur Blindsight :
  post-Sprint 12. Demande un Blindsight fini + un run multi-seeds.
- Runs Phase F sur Narval : post-sprint 16.

---

## Risques identifiés

- **R1** — Le test parity `cae_loss` vs paper_reference peut échouer
  pour des raisons subtiles (ordre des opérations float, init RNG state
  différent). À traiter en utilisant les helpers `external/paper_reference/`
  directement (importer la fonction, pas la réécrire dans le test).
- **R2** — `retain_graph=True` à travers 50 unrolls peut faire OOM sur
  certaines machines pour le test d'intégration. Solution : batch_size=2
  dans le test, ou détacher périodiquement (mais alors plus paper-faithful
  → flag dans le test).
- **R3** — D11.5 = delete : SARL+CL Sprint 15 utilise
  `weight_regularization` (pas `distillation_loss`) donc impact nul à
  court terme. Si futur besoin KL distillation : récupérer depuis
  `git show main:src/maps/components/losses.py`.
- **R4** — D11.7 SimCLR sans référence parity : les tests sont sanity
  uniquement. Risque que la maths NT-Xent soit subtilement fausse
  (signe, axis de softmax, exclusion de la self-similarity dans le
  denominator) sans qu'on le détecte avant Sprint 12 comparison.
  Mitigation : tester contre 2-3 implémentations de référence publiques
  (e.g. `lightly`, `pytorch-metric-learning`) en plus des sanity checks.

---

## Next sprint (Sprint 12 — préparation)

Sprint 11 produit un `core/` testé (avec `simclr_loss` math fini).
Sprint 12 ouvre **Blindsight** et ajoute :

- `networks/first_order_mlp.py` (shared avec AGL)
- `utils/{config,logging_setup,device}.py` (le reste de utils)
- `domains/blindsight/{data.py, trainer.py, cli.py}`
- `domains/blindsight/augmentations.py` — **point clé D11.7** :
  augmentations stimulus-binaires pour SimCLR positive pairs (bit-flip
  p%, dropout d'éléments, etc.). Choix de recherche à acter Sprint 12
  Day 1.
- Tests parity Blindsight contre `external/paper_reference/blindsight_tmlr.py`
- Trainer Blindsight branchable sur `first_order_loss.kind ∈ {cae, simclr}`

Si le `core/` est solide, le risque Sprint 12 baisse à
"reproductibilité des chiffres Blindsight" pur — pas de risque
mathématique sur les composants partagés. **La comparaison empirique
CAE vs SimCLR** sur Blindsight devient possible post-Sprint 12, et
constitue un livrable scientifique à part entière (potentiellement
un report `docs/reports/cae-vs-simclr-blindsight.md`).
