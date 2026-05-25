# CLAUDE.md — MAPS Project

## Project Context

**MAPS** = Metacognitive Architecture for Perceptual and Social Learning (Vargas, Kastel, Pasquali, Cleeremans, Sheikhbahaee, Dumas — TMLR submission).

Two components added to a baseline network:
1. **Second-order network** — comparator matrix + 2 wagering units (Pasquali & Cleeremans 2010, "Know Thyself")
2. **Cascade model** — graded activation accumulation over iterations (McClelland 1989), α=0.02, 50 iterations

Tested on 4 domains (2×2 factorial on/off → 6 settings per paper):
- **Blindsight** — perceptual detection under noise
- **AGL** — Artificial Grammar Learning (implicit/explicit)
- **SARL** — MinAtar DQN (Space Invaders, Breakout, Seaquest, Asterix, Freeway)
- **SARL+CL** — Continual learning with teacher network
- **MARL** — MeltingPot 2.0 MAPPO (Commons Harvest, Chemistry, Territory)
- **METTA-AI** — exploratory extension (separate env, external/)

**Lab:** PPSP, CHU Sainte-Justine. **Maintainer (fork):** Rémy Ramadour. **Original authors:** Juan David Vargas et al.

---

## Current Status (Sprint 11 ✅ closed — core rewritten ; Sprint 12 next)

This is a fork being cleaned up for reproducibility after the original student left. Sprints 00-09 produced a modular, config-first port. **Sprint 10** wiped `src/maps/` and produced 41 reverse-prompts (one per module). **Sprint 11** has rebuilt `core/` (cascade + second_order + losses, ~700 LOC, 67/67 tests pass including 5 parity vs paper_reference).

**Branches:**
- `refactor/core` — current head. Core/ written + tested.
- `refactor/main-rewrite` — Sprint 10 baseline (empty src/maps/).
- `main` — Sprint 09 state, kept as numerical reference.

**Sprint 12 (next) :** Blindsight. Adds `networks/first_order_mlp.py`, the rest of `utils/`, and `domains/blindsight/{data,trainer,cli,augmentations}.py`. Will exercise the `core/` API end-to-end and validate paper-reproduction z-scores.

See `docs/sprints/sprint-11-core-rewrite.md` for Sprint 11 closeout, `docs/learning/walkthroughs/cascade-from-paper-to-code.md` for the cascade pedagogy, and `docs/learning/structure-decision.md` for the layout decisions taken in Sprint 10.

---

## Canonical MAPS Constants (from paper)

These are **locked** — any deviation must be justified and logged.

| Parameter | Value | Source |
|-----------|-------|--------|
| `cascade_alpha` (α) | 0.02 | Paper §2.1 eq.6 |
| `cascade_iterations` | 50 | Paper §2.1 ("empirically selected") |
| `first_order_hidden_dim` | 40 | AGL default |
| `second_order_hidden_dim` | 100 | AGL default |
| `wagering_units` | 2 | Koch & Preuschoff 2007 |
| `ema_window` | 25 | SARL shell default |
| `scheduler_step` | 25 | Blindsight/AGL StepLR |
| `scheduler_gamma` | 0.98 | Blindsight/AGL StepLR |
| `default_optimizer` | ADAMAX | Blindsight/AGL default |
| `RANDOM_SEED` (default) | 42 | Lab-wide convention |

All of these live in `config/maps.yaml`. **Do not hardcode them in training scripts.**

---

## Target Folder Layout (Sprint 10+ structure)

```
src/maps/                # package — empty during Sprint 10, populated Sprint 11+
├── __init__.py
├── core/                # cascade.py, second_order.py, losses.py — shared math
├── networks/            # first_order_mlp.py (shared Blindsight/AGL MLP)
├── domains/             # one sub-package per evaluation domain
│   ├── blindsight/      # data.py, trainer.py, cli.py
│   ├── agl/             # data.py, pool.py, trainer.py, cli.py
│   ├── sarl/            # data.py, model.py (v2), model_v1.py, training_loop.py,
│   │                    #   trainer.py, actor_critic.py, cli.py
│   ├── sarl_cl/         # model.py, loss_weighting.py, training_loop.py,
│   │                    #   trainer.py, cli.py
│   ├── marl/            # runner.py, trainer.py, policy.py, env.py, data.py,
│   │                    #   encoder.py, act.py, rnn.py, valuenorm.py, setting.py,
│   │                    #   util.py, cli.py
│   └── metta/           # last (exploratory)
├── utils/               # config.py, logging_setup.py, seeding.py, device.py,
│                        #   energy_tracker.py
└── cli/                 # optional global Typer app dispatching to each domain

config/                  # YAML config (Sprint 11.5 reorg D-cleanup.2 Option C)
├── maps.yaml            # canonical paper constants (locked)
├── paths.yaml           # filesystem path helpers
├── domains/             # per-domain (env + training)
│   ├── blindsight/      # env.yaml, training.yaml
│   ├── agl/             # training.yaml
│   ├── sarl/            # training.yaml, acb.yaml
│   ├── sarl_cl/         # training.yaml
│   └── marl/            # env/{*.yaml}, training.yaml
└── experiments/         # factorial_{2x2,6cell,marl}.yaml

external/                # vendored: METTA, MinAtar, paper_reference (numerical truth)
tests/                   # unit + parity (Sprint 11+ rewrite per-domain)
docs/
├── sprints/             # current sprint specs (00-09 historical, on main)
├── reproduction/        # deviations.md, experiment_matrix.md, paper audits
├── reviews/             # technical reviews (Sprint-08 C.* sub-phases — kept as reference)
├── reports/             # per-sprint closeouts (most historical, on main)
├── figures/             # paper figures (Sprint 11.5: ex-/images/)
├── pdf/                 # reference papers (Sprint 11.5: ex-/pdf/)
└── learning/            # Sprint 10 pedagogical layer
    ├── structure-decision.md
    ├── glossaire.md       # vocabulaire MAPS (Sprint 11.6)
    ├── reverse-prompts/   # one .md per module, mirror src/maps/ path
    ├── walkthroughs/      # narrative post-refactor
    └── chat-prompts/      # prompts ready to paste in Claude chat (web)

notebooks/learning/      # executable tutorials (Jupyter)
outputs/ logs/ models/ data/   # gitignored
```

**Naming conventions** :
- `from __future__ import annotations` in every module
- Imports absolute (`from maps.core.cascade import ...`)
- NumPy-style docstrings with inline paper citations
- Greek letters allowed (α, β, γ, λ — ruff RUF001/002/003 ignored)

---

## Tech Stack

- **Python 3.12** via `uv` (single `pyproject.toml` at root, `uv.lock` committed)
- **Linting:** `ruff check`, `ruff format` | **Testing:** `pytest`
- **Config:** `OmegaConf`/Hydra-compatible YAML
- **Logging:** stdlib `logging` (no `print`, no silenced progress bars)
- **Git:** conventional commits, scoped (`feat(sarl): ...`, `fix(cascade): ...`)

### Optional dependency groups

```toml
[project.optional-dependencies]
blindsight = ["torch", "torchmetrics", ...]
agl        = ["torch", ...]
sarl       = ["torch", "gym", ...]        # requires local MinAtar install
marl       = ["torch", "ray", ...]        # LINUX ONLY (dmlab2d, meltingpot)
metta      = []                           # handled in external/METTA (own env)
dev        = ["pytest", "ruff", "black", "pre-commit", "detect-secrets"]
```

### Default commands

```bash
uv sync --extra blindsight --extra agl --extra sarl --extra dev
uv run pytest                                     # all tests
uv run pytest tests/reproduction -m slow          # reproduce paper z-scores
uv run ruff check . && uv run ruff format --check
```

Or via Makefile (Sprint 11.6, pattern emprunté à bilevel-fishery) :

```bash
make install      # uv sync + pre-commit install
make test-core    # Sprint 11 core/ tests only
make test         # full suite
make check        # lint + format-check + test
make clean        # purge .pyc / __pycache__ / .DS_Store / caches
make help         # discoverable list of targets
```

---

## Compute Targets

| Domain | Mac M-series (local) | Compute Canada Narval/Rorqual |
|--------|----------------------|--------------------------------|
| Blindsight | ✅ CPU/MPS | ✅ (not needed) |
| AGL | ✅ CPU/MPS | ✅ (not needed) |
| SARL (MinAtar) | ✅ CPU | ✅ GPU for speed |
| SARL+CL | ✅ CPU | ✅ GPU |
| MARL (MeltingPot) | ❌ Linux-only deps | ✅ GPU required |
| METTA | ⚠️ partial | ✅ GPU required |

---

## Scientific Rigor Rules

- Always cite source papers in docstrings
- Flag heuristic vs. theoretically grounded choices
- Seed control is **mandatory** — `maps.utils.seeding.set_all_seeds(seed)` at every entry point
- Re-run numbers must come with CI or seed std, never single-seed point estimates
- Never silently change hyperparameters from paper — if needed, log in `docs/reproduction/deviations.md`

---

## File References

| File | Role |
|------|------|
| `docs/TODO.md` | Technical debt register |
| `docs/sprints/sprint-10-reappropriation-phase-0-and-1.md` | Current sprint spec |
| `docs/learning/structure-decision.md` | Layout decisions Sprint 10 Phase 0 |
| `docs/learning/glossaire.md` | Vocabulaire MAPS (Sprint 11.6) |
| `Makefile` | Convenience targets — `make help` to discover |
| `docs/learning/reverse-prompts/` | Reverse-prompt per Python module (Phase 1) |
| `docs/reproduction/deviations.md` | 52 paper↔code deviations tracked |
| `docs/reproduction/experiment_matrix.md` | Paper z-score targets |
| `config/maps.yaml` | Canonical MAPS component constants |
| `docs/pdf/MAPS_TMLR_Journal_Submission.pdf` | Full paper (reference) |
| `docs/28_11_25.pdf` | Project direction notes (Guillaume meeting) |
| `external/paper_reference/` | Original Vargas code (44k LOC), numerical reference |
