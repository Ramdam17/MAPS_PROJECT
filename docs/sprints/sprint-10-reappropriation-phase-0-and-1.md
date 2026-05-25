# Sprint 10 — Réappropriation : Phase 0 (structure) + Phase 1 (reverse-prompts)

**Status:** 🟢 in-progress
**Branch:** `refactor/main-rewrite` (créée 2026-05-25 depuis `main` @ 6b200ab)
**Owner:** Rémy Ramadour
**Estimated effort:** ~5-10 sessions Claude (Phase 0 décisions + Phase 1 reverse-prompts ~25 modules)
**Depends on:** méta-plan approuvé (`~/.claude/plans/yo-j-aimerais-que-abstract-sutton.md`)

---

## Context

Le projet a traversé 9 sprints de refactor technique (Sprints 00-09) qui ont
produit une codebase modulaire, config-first, avec tests parity tier 1/2/3.
Mais Rémy n'a pas écrit ce code — il en a hérité. Pour la suite (défense
TMLR, réponses Natalie/Guillaume, maintenance long terme), il faut une
**réappropriation intellectuelle**, pas un nouveau refactor technique.

Ce sprint démarre cette réappropriation par deux étapes méta (avant tout
code) :

- **Phase 0** : décider ENSEMBLE la structure idéale du repo (dossiers,
  fichiers, conventions). Avant d'écrire la première ligne neuve.
- **Phase 1** : reverse-prompter chaque module du code actuel — produire
  un `.md` qui répond "quelle aurait été la spec qu'on aurait donné à
  Claude pour produire ce code ?". Outil pédagogique central.

Aucun code Python n'est écrit ni modifié pendant ce sprint. Seules des
décisions et de la documentation sont produites. Le code actuel sur
`refactor/main-rewrite` reste **inchangé** : c'est le matériau qu'on
analyse, pas qu'on transforme. La transformation viendra aux sprints
11-15 (un par domaine).

## Open items (entry conditions)

1. **Niveau de "from-scratch"** : à trancher avec Rémy. Branche
   `refactor/main-rewrite` créée comme copie exacte de `main`. Question :
   pendant Phase 0/1, on touche-t-on à `src/maps/`, `scripts/`,
   `config/` ? Ma proposition : **non**, on laisse tout en l'état. Le
   "from-scratch" commence sprint 11 (Blindsight) où on remplace
   `src/maps/experiments/blindsight/` par une nouvelle implémentation.

2. **Squelette de structure** : à décider en Phase 0. Voir strawman
   ci-dessous.

3. **Format reverse-prompts** : un fichier `.md` par module Python, en
   `docs/learning/reverse-prompts/<chemin-module>.md`. À valider.

4. **Conventions** : naming, imports, tests, configs. À discuter.

## Goals (DoD)

### Phase 0 — Décider la structure (Sessions 1-2)

- [ ] **Décision 0.1** : niveau "from-scratch" (rien touché Phase 0/1, ou
      delete partiel ?)
- [ ] **Décision 0.2** : layout `src/maps/` (3 propositions ci-dessous +
      ouvert à autres)
- [ ] **Décision 0.3** : sort de `scripts/` (gardé tel quel, fusionné
      dans `src/maps/cli/`, ou autre ?)
- [ ] **Décision 0.4** : layout doc pédagogique (`docs/learning/` proposé)
- [ ] **Décision 0.5** : format reverse-prompts (1 .md par module
      proposé)
- [ ] **Décision 0.6** : conventions (naming, imports, tests, configs)
- [ ] **Livrable** : `docs/learning/structure-decision.md` qui acte tous
      les choix avec justification
- [ ] **Livrable** : squelette dossiers vide committé sur
      `refactor/main-rewrite` (si Décision 0.1 = delete partiel)

### Phase 1 — Reverse-prompts par module (Sessions 3-N)

Pour chaque module Python du code actuel, produire
`docs/learning/reverse-prompts/<module>.md` répondant à :

1. **Quelle aurait été la spec / le prompt** qu'on aurait donné à
   Claude pour produire ce code ?
2. **Contraintes scientifiques** : équations du paper qui forcent cette
   forme, choix d'hyperparams cités dans la littérature.
3. **Contraintes d'ingénierie** : ce qui force des choix de design
   (paramétrer, tester, accélérer, modularité, type hints).
4. **Deviations / dettes présentes** : ce qui dévie du paper ou est
   sous-optimal (croiser avec `docs/reproduction/deviations.md` et
   `docs/TODO.md`).
5. **Questions ouvertes** : ce qu'on ne comprend pas en lisant.

**Modules à couvrir** (~25 fichiers, ordre proposé) :

**1. Components partagés (3 modules)** :
- `src/maps/components/cascade.py`
- `src/maps/components/second_order.py`
- `src/maps/components/losses.py`

**2. Networks partagés (1 module)** :
- `src/maps/networks/first_order_mlp.py` (ou équivalent)

**3. Utils (4 modules)** :
- `src/maps/utils/config.py`
- `src/maps/utils/logging_setup.py`
- `src/maps/utils/seeding.py`
- `src/maps/utils/energy_tracker.py`

**4. Blindsight (3 modules)** :
- `src/maps/experiments/blindsight/data.py`
- `src/maps/experiments/blindsight/trainer.py`
- `scripts/run_blindsight.py`

**5. AGL (4 modules)** :
- `src/maps/experiments/agl/data.py`
- `src/maps/experiments/agl/pool.py`
- `src/maps/experiments/agl/trainer.py`
- `scripts/run_agl.py`

**6. SARL (7 modules)** :
- `src/maps/experiments/sarl/data.py`
- `src/maps/experiments/sarl/model.py` (v2)
- `src/maps/experiments/sarl/model_v1.py`
- `src/maps/experiments/sarl/training_loop.py`
- `src/maps/experiments/sarl/trainer.py`
- `src/maps/experiments/sarl/actor_critic.py` (ACB Setting 7)
- `scripts/run_sarl.py`

**7. SARL+CL (4 modules)** :
- `src/maps/experiments/sarl_cl/model.py`
- `src/maps/experiments/sarl_cl/training_loop.py`
- `src/maps/experiments/sarl_cl/loss_weighting.py`
- `scripts/run_sarl_cl.py`

**8. MARL (~6 modules)** :
- `src/maps/experiments/marl/runner.py`
- `src/maps/experiments/marl/trainer.py`
- `src/maps/experiments/marl/policy.py`
- `src/maps/experiments/marl/env.py`
- `src/maps/experiments/marl/encoder.py`
- `scripts/run_marl.py`

**Total: ~32 modules** (légèrement plus que mon estimation initiale de 25).
À voir si on veut tout traiter ou en grouper certains.

### Done when

- `docs/learning/structure-decision.md` écrit, validé, committé.
- (si applicable) Squelette dossiers committé.
- Tous les ~32 modules ont leur reverse-prompt en
  `docs/learning/reverse-prompts/`.
- Sprint doc 11 (Blindsight) prêt à être ouvert avec une vision claire
  de la structure cible.

---

## Strawman — Structure proposée (Phase 0)

### Layout `src/maps/` — Option A (conservative)

```
src/maps/
├── __init__.py
├── core/                    # ← renamed from components/ (clearer name)
│   ├── cascade.py
│   ├── second_order.py
│   └── losses.py
├── networks/                # shared architectures
│   └── first_order_mlp.py
├── domains/                 # ← renamed from experiments/ (clearer)
│   ├── blindsight/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── trainer.py
│   │   └── cli.py           # ← co-located CLI (replaces scripts/run_*.py)
│   ├── agl/
│   ├── sarl/
│   ├── sarl_cl/
│   ├── marl/
│   └── metta/               # last, exploratory
├── utils/
│   ├── config.py
│   ├── logging_setup.py
│   ├── seeding.py
│   ├── device.py            # ← NEW: auto-detect CPU/MPS/CUDA
│   └── energy_tracker.py    # ← may be moved into domains/marl/
└── cli/                     # global CLI entry (alternative à scripts/)
    └── main.py              # `maps blindsight ...`, `maps agl ...`
```

**Changements vs. structure actuelle** :
- `components/` → `core/` (nom plus parlant)
- `experiments/` → `domains/` (cohérent avec le vocabulaire paper)
- `scripts/run_<domain>.py` → `src/maps/domains/<domain>/cli.py` +
  optionnel `src/maps/cli/main.py` unifié
- Ajout `src/maps/utils/device.py` (auto-detection MPS/CUDA/CPU)
- Préparation espace `domains/metta/` pour sprint final

### Layout `docs/` — Proposition

```
docs/
├── sprints/                 # inchangé
├── reports/                 # inchangé
├── reproduction/            # inchangé (deviations.md, etc.)
├── reviews/                 # inchangé
├── plans/                   # inchangé
├── learning/                # NEW — pédagogique
│   ├── structure-decision.md
│   ├── reverse-prompts/
│   │   ├── core/
│   │   │   ├── cascade.md
│   │   │   ├── second_order.md
│   │   │   └── losses.md
│   │   ├── networks/
│   │   ├── utils/
│   │   └── domains/
│   │       ├── blindsight/
│   │       ├── agl/
│   │       └── ...
│   ├── walkthroughs/         # narrative post-refactor
│   │   ├── cascade-from-paper-to-code.md
│   │   └── ...
│   └── chat-prompts/         # prompts à coller dans Claude chat (web)
│       ├── understanding-cascade.md
│       └── ...
└── TODO.md                   # inchangé
```

### Top-level (racine repo) — Proposition

```
.
├── src/maps/                # comme ci-dessus
├── tests/                   # INCHANGÉ — parity tier 1/2/3 = gardiens
├── external/                # INCHANGÉ — paper_reference, METTA, MinAtar
├── config/                  # INCHANGÉ pour Phase 0/1 ; à revoir Phase 3
├── docs/                    # comme ci-dessus
├── scripts/                 # ← DELETED si on adopte domains/<x>/cli.py
├── notebooks/               # gardé pour notebooks expérimentaux
├── outputs/ logs/ models/ data/   # gitignored, inchangé
├── pdf/                     # inchangé
├── pyproject.toml           # gardé, ajustements ruff/black au passage
├── .pre-commit-config.yaml  # gardé
├── README.md                # à réécrire en sprint final
└── CLAUDE.md                # à enrichir au passage
```

### Conventions proposées

- **Modules** : `snake_case.py`
- **Classes** : `PascalCase`
- **Fonctions** : `snake_case`, type hints obligatoires (`from __future__
  import annotations` en tête)
- **Lettres grecques** : autorisées (α, β, γ, λ — déjà acquis ruff)
- **Imports** : absolus par défaut (`from maps.core.cascade import ...`),
  relatifs uniquement pour sous-modules d'un même package
- **Docstrings** : NumPy-style, citations papier inline (ex. "Pasquali
  & Cleeremans (2010)"), explication WHY pour choix non-obvious
- **Logging** : `logger = logging.getLogger(__name__)` en tête, jamais
  `print()` en module (OK dans `cli.py`)
- **Tests** : `tests/parity/<domain>/` pour parity, `tests/unit/<package>/`
  pour unitaires, `tests/integration/` pour smoke
- **Configs** : un YAML par domaine (`config/domains/<domain>.yaml`),
  un YAML global (`config/maps.yaml`), un YAML par expérience
  factorielle (`config/experiments/*.yaml`)

---

## Phases (questions et décisions)

| Phase | Date cible | Décisions à prendre |
|-------|------------|---------------------|
| 0.A — Niveau from-scratch + Layout `src/maps/` | Session 1 | Décisions 0.1, 0.2 |
| 0.B — scripts/cli + doc + conventions | Session 2 | Décisions 0.3, 0.4, 0.5, 0.6 |
| 0.C — `structure-decision.md` consolidé | Session 2 | Livrable Phase 0 |
| 1.A — Reverse-prompts `core/` (3 modules) | Sessions 3-4 | cascade, second_order, losses |
| 1.B — Reverse-prompts `networks/` + `utils/` | Sessions 4-5 | first_order_mlp, config, logging, seeding, energy |
| 1.C — Reverse-prompts Blindsight (3) | Session 6 | data, trainer, cli |
| 1.D — Reverse-prompts AGL (4) | Session 7 | data, pool, trainer, cli |
| 1.E — Reverse-prompts SARL (7) | Sessions 8-9 | data, model v1+v2, training_loop, trainer, actor_critic, cli |
| 1.F — Reverse-prompts SARL+CL (4) | Session 10 | model, training_loop, loss_weighting, cli |
| 1.G — Reverse-prompts MARL (6) | Sessions 11-12 | runner, trainer, policy, env, encoder, cli |
| 1.H — Consolidation + transition vers Sprint 11 | Session 13 | Synthèse, ouvrir sprint-11-blindsight.md |

---

## Garde-fous

- **Aucun code Python touché** pendant ce sprint. Si on a envie de
  toucher au code, on stoppe et on note pour Sprint 11+.
- **Tests parity restent verts** sur `refactor/main-rewrite` (puisque
  rien n'est modifié dans le code).
- **Reverse-prompts honnêtes** : si on ne comprend pas, on le dit
  explicitement dans les "Questions ouvertes".

## Hors scope (ce sprint)

- Refactor du code (sprints 11-15)
- Optimisation perfs (sprints 11-15)
- Resolution dettes DETTE-1..4 (sprints 11-15)
- Runs Phase F sur Narval (post-sprint 15)
- Mise à jour TMLR / Natalie (à part)
