# Structure Decision — Sprint 10 Phase 0 (2026-05-25)

**Status :** ✅ acted on branch `refactor/main-rewrite`
**Decided with :** Rémy Ramadour (interactive session)
**Méta-plan :** `~/.claude/plans/yo-j-aimerais-que-abstract-sutton.md`
**Sprint doc :** `docs/sprints/sprint-10-reappropriation-phase-0-and-1.md`

This document records the structural decisions taken at the start of the
MAPS rewrite. Each decision has a **Why** (rationale) and a **How to
apply** (how it propagates into the rebuild).

---

## Décision 0.1 — Niveau "from-scratch"

**Choix : suppression entière de `src/maps/` et `scripts/` maintenant.**

The previous implementation (Sprints 00-09, 12 kLOC src + 2 kLOC scripts)
has been removed from this branch. It remains accessible on `main` and in
git history (e.g. `git show main:src/maps/components/cascade.py`).

**Why :** Rémy wants a real "blank-slate" rewrite, not a refactor. Keeping
the old code under his nose would be a temptation to copy-translate rather
than re-think. The reverse-prompts (Phase 1) will pull from `main` as
needed without polluting the working tree.

**How to apply :**
- During Sprint 10 (Phase 0/1) : no Python is written. Only docs and
  scaffolding `__init__.py`.
- During Sprint 11+ : each domain rebuilds its module set from the
  reverse-prompts + the paper, not by copying the deleted code.
- The numerical reference for parity is `external/paper_reference/`
  (the original Vargas code, 44 kLOC, verbatim) — not the Sprint 09
  implementation. This is more legitimate because the paper Tables 5-7
  numbers were produced by that original code.
- `tests/parity/` currently has broken imports (`from maps.*`). They will
  be rewritten per-domain in Sprint 11+ to compare against
  `external/paper_reference/`.

---

## Décision 0.2 — Layout `src/maps/` (Option A : Conservative + renommages)

**Choix : Option A.**

```
src/maps/
├── __init__.py
├── core/                    # ← renamed from components/
│   ├── cascade.py           # McClelland (1989) cascade model
│   ├── second_order.py      # Pasquali & Cleeremans (2010) comparator + wager
│   └── losses.py            # CAE + wagering BCE + distillation
├── networks/                # shared architectures (Blindsight + AGL MLP)
│   └── first_order_mlp.py
├── domains/                 # ← renamed from experiments/
│   ├── blindsight/
│   │   ├── __init__.py
│   │   ├── data.py          # stimulus conditions, dataset construction
│   │   ├── trainer.py       # training/evaluation loop
│   │   └── cli.py           # Typer CLI — `python -m maps.domains.blindsight.cli`
│   ├── agl/
│   ├── sarl/                # v1 canonical (Sprint 09) + actor-critic baseline
│   ├── sarl_cl/             # continual learning + teacher distillation
│   ├── marl/                # MeltingPot 2.0 MAPPO
│   └── metta/               # last (exploratory)
├── utils/
│   ├── config.py            # OmegaConf YAML loader + CLI overrides
│   ├── logging_setup.py     # stdlib logging configuration helper
│   ├── seeding.py           # set_all_seeds(seed) — random, numpy, torch, cuda
│   ├── device.py            # NEW: auto-detect CPU / MPS / CUDA
│   └── energy_tracker.py    # may move to maps.domains.marl (Sprint 15, DETTE-4)
└── cli/
    └── main.py              # optional top-level Typer app dispatching to domains
```

**Why :**
- `components/ → core/` : `core` reads as "load-bearing infrastructure",
  not "reusable widgets".
- `experiments/ → domains/` : aligns with the paper's vocabulary
  (Blindsight / AGL / SARL / MARL are "domains", not "experiments").
- Co-located `cli.py` per domain : reduces context-switching when
  reading a domain (data + model + training + CLI in one folder).
- New `utils/device.py` : current code doesn't auto-detect MPS for
  Apple Silicon users (Rémy works on M-series locally). One-line fix
  that improves DX.
- Optional `cli/main.py` : convenience entry. The per-domain CLIs are
  the canonical interface ; the global one just dispatches.

**How to apply :** every domain rebuild (Sprint 11+) lands its `data.py`,
`trainer.py`, `cli.py` inside `src/maps/domains/<domain>/`. Shared math
goes into `src/maps/core/`. Shared architectures go into
`src/maps/networks/`. Utility helpers (cross-domain) go into
`src/maps/utils/`.

---

## Décision 0.3 — Sort de `scripts/` (implicite : supprimé)

**Choix : suppression. Pas de répertoire `scripts/` au top-level.**

Les anciens entry points `scripts/run_<domain>.py` sont remplacés par
`src/maps/domains/<domain>/cli.py` co-localisés. Les SLURM submission
scripts (`scripts/slurm/*.sh`) sont aussi supprimés ; ils seront
réécrits/réintroduits lors de la préparation des runs Narval/Tamia
(post-sprint 15, hors scope du refactor).

**Why :** un endroit unique pour le code d'un domaine = moins de
context-switching, moins de risque d'oublier de propager un changement.

**How to apply :** Sprint 11+ écrit le CLI au même endroit que le
trainer.

---

## Décision 0.4 — Doc pédagogique (Mix `docs/learning/` + `notebooks/learning/`)

**Choix : Mix.**

```
docs/learning/
├── structure-decision.md             # ce fichier
├── reverse-prompts/                  # Phase 1
│   ├── core/
│   │   ├── cascade.md
│   │   ├── second_order.md
│   │   └── losses.md
│   ├── networks/
│   │   └── first_order_mlp.md
│   ├── utils/
│   │   ├── config.md
│   │   ├── logging_setup.md
│   │   ├── seeding.md
│   │   └── energy_tracker.md
│   └── domains/
│       ├── blindsight/
│       │   ├── data.md
│       │   ├── trainer.md
│       │   └── cli.md
│       ├── agl/
│       ├── sarl/
│       ├── sarl_cl/
│       ├── marl/
│       └── metta/
├── walkthroughs/                     # narratif post-refactor (Sprint 11+)
│   └── <topic>.md                    # ex. cascade-from-paper-to-code.md
└── chat-prompts/                     # prompts à coller dans Claude chat web
    └── <topic>.md

notebooks/learning/                   # tutoriels exécutables
└── <NN_topic>.ipynb
```

**Why :**
- Markdown reverse-prompts + walkthroughs → **searchable, archivable,
  citable, git-friendly**. Rémy relira ces docs dans 6 mois.
- Notebooks Jupyter pour tutoriels où il faut **exécuter et voir le
  résultat** : visualiser une cascade qui converge, tracer une courbe
  d'apprentissage, etc. Outils différents pour besoins différents.
- `chat-prompts/` : préparation de prompts à copier-coller dans Claude
  chat (web) avec contexte, pour creuser un concept en parallèle de la
  session Claude Code. Pattern de workflow demandé explicitement.

**How to apply :**
- Reverse-prompts (Sprint 10 Phase 1) : un `.md` par module Python.
- Walkthroughs (Sprint 11+) : un `.md` narratif par grand sujet
  pédagogique, écrit *après* avoir compris (post-mortem digest).
- Notebooks (selon besoin) : quand l'exécution / visualisation
  apporte vraiment quelque chose. Pas par défaut.
- `chat-prompts/` : un `.md` quand on identifie une question qui mérite
  d'être posée à Claude chat avec contexte structuré.

---

## Décision 0.5 — Format reverse-prompts (1 .md par module Python)

**Choix : un fichier markdown par module Python, chemin miroir.**

`src/maps/core/cascade.py` → `docs/learning/reverse-prompts/core/cascade.md`
`src/maps/domains/blindsight/trainer.py` →
  `docs/learning/reverse-prompts/domains/blindsight/trainer.md`

**Template de chaque reverse-prompt :**

```markdown
# Reverse-prompt — <module path>

## 1. Quelle aurait été la spec ?

[Comme si on disait à Claude : "écris un module qui fait ceci...".
Niveau d'abstraction : intention, contraintes, interfaces.]

## 2. Contraintes scientifiques

[Équations du paper référencées, hyperparams cités dans la littérature,
choix imposés par la méthode (ex. ADAMAX pour Blindsight/AGL).]

## 3. Contraintes d'ingénierie

[Modularité, type hints, logging, paramétrabilité, perf, tests.]

## 4. Deviations / dettes présentes

[Croiser avec docs/reproduction/deviations.md et docs/TODO.md.
Quelles D-* sont visibles dans ce module ? Quelles DETTE-* ?
Quels hardcodes ?]

## 5. Questions ouvertes

[Ce qu'on ne comprend pas en lisant. À discuter, à creuser, à demander
à Claude chat.]

## 6. Pour le rewrite (Sprint 11+)

[Notes d'intention pour la réécriture : que garder, que changer, où
mettre les nouvelles docstrings/comments WHY, quelles opportunités
perfs identifiées dans ce module précisément.]
```

**Why :** une granularité par module = sessions Claude Code digestes,
chaque doc indépendamment relisable. Le template force à séparer le
"comment c'est fait" du "comment je le ferais".

**How to apply :** Phase 1 produit ~32 fichiers selon ce template.

---

## Décision 0.6 — Conventions de code

**Choix : 4 conventions actées (toutes sélectionnées).**

### Imports absolus par défaut

```python
# OUI
from maps.core.cascade import cascade_update
from maps.utils.seeding import set_all_seeds

# OK pour sous-modules d'un même package
# (dans maps/domains/blindsight/trainer.py)
from .data import build_dataset

# NON (pas de chemins relatifs au-delà d'un niveau)
from ...core.cascade import cascade_update
```

**Why :** absolus = grep-friendly, refactor-friendly (pas besoin de
recalculer les `..` quand on déplace un fichier), explicites.

### NumPy-style docstrings + citations papier inline

```python
def cascade_update(
    new_activation: Tensor,
    prev_activation: Tensor | None,
    cascade_rate: float,
) -> Tensor:
    """One iteration of the McClelland cascade.

    The cascade model integrates new evidence into a running activation
    state at rate ``α`` (typically 0.02). After ~50 iterations the
    activation converges to the steady-state response of the network.

    Parameters
    ----------
    new_activation : torch.Tensor
        Output of the network's forward pass at this iteration.
    prev_activation : torch.Tensor or None
        Activation from the previous iteration, or ``None`` at t=0.
    cascade_rate : float
        Integration rate α ∈ (0, 1]. Paper §2.1 eq.6 fixes α = 0.02.

    Returns
    -------
    torch.Tensor
        Updated activation ``α · new + (1 - α) · prev``.

    Notes
    -----
    Why α = 0.02, 50 iterations? See [McClelland1989]_ for the cascade
    formulation and [Vargas2025]_ §2.1 for the MAPS-specific tuning.

    References
    ----------
    .. [McClelland1989] McClelland, J. L. (1989). Cascade models...
    .. [Vargas2025] Vargas et al. (2025). MAPS, TMLR submission.
    """
```

**Why :** NumPy-style is the lab standard (`hypyp`, `mne`, `scipy` all
use it). Citations inline forcent à *savoir* d'où vient une décision,
pas juste à la copier.

### `from __future__ import annotations` en tête

```python
# every module starts with:
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# annotations stay as strings — no runtime cost, support forward refs
def train(cfg: TrainConfig) -> TrainResult: ...
```

**Why :** PEP 563, Python 3.12 fait pas encore ça par défaut, recommandé
pour les codebases modernes. Permet d'écrire `Tensor | None` partout
sans coût runtime, et de gérer les forward refs proprement.

### Greek letters autorisées (α, β, γ, λ)

`ruff` ignore `RUF001/002/003` dans `pyproject.toml`. On écrit
`α = 0.02` plutôt que `cascade_rate = 0.02` quand on est dans une formule
mathématique du paper, et `cascade_rate` quand on est dans une interface
publique.

**Why :** lisibilité côté chercheur. Le code se rapproche du paper.

---

## Conséquences immédiates

1. **`pyproject.toml`** : nettoyage des références à des fichiers
   supprimés (per-file-ignores SARL/SARL+CL/MARL, coverage omits,
   sdist include `scripts`). À faire dans le commit Phase 0.
2. **`.pre-commit-config.yaml`** : la règle `ban-abs-home-paths` filtre
   sur `^(src/|scripts/|tests/|config/)` — `scripts/` n'existe plus mais
   le pattern ne fait pas mal. Garder tel quel pour éviter du churn.
3. **`tests/parity/`** : imports cassés. **Non-bloquant** pendant
   Sprint 10 (on ne run pas pytest). À reconstruire au Sprint 11+ par
   domaine.
4. **CLAUDE.md** (project) : section "Target Folder Layout" mentionne
   l'ancienne structure (`experiments/`, `scripts/`). À mettre à jour
   en fin de Sprint 10 quand la nouvelle structure est éprouvée.

## Hors scope (explicitement)

- Réécriture du code Python : Sprint 11+.
- Modification de `config/*.yaml` : à voir Sprint 11+ (probablement
  un nettoyage, mais pas une refonte).
- Suppression de `external/paper_reference/` ou `external/MinAtar/` :
  jamais (références numériques + deps).
- Suppression de `external/METTA/` : à voir Sprint 16 (DETTE-39, peut
  devenir un submodule).
