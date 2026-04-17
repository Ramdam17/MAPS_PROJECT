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

## Current Status (Sprint 00 — Remise en route)

This is a fork being cleaned up for reproducibility after the original student left. See `docs/sprints/sprint-00-remise-en-route.md`.

**Branch:** `refactor/clean-rerun`

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

## Target Folder Layout

```
src/maps/                # core package (components, networks, training, evaluation, utils, experiments)
scripts/                 # CLI entry points (typer)
config/                  # YAML (maps.yaml, paths.yaml, training/*, env/*, experiments/*)
notebooks/               # NN_snake_case.ipynb tutorials
tests/                   # unit + numerical + reproduction
external/                # vendored: METTA, MinAtar, meltingpot
docs/                    # sprints, reports, specifications, reproduction
outputs/ logs/ models/ data/   # gitignored
pdf/                     # reference papers
```

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
| `docs/sprints/sprint-00-remise-en-route.md` | Current sprint spec |
| `docs/reproduction/experiment_matrix.md` | Paper z-score targets |
| `config/maps.yaml` | Canonical MAPS component constants |
| `pdf/MAPS_TMLR_Journal_Submission.pdf` | Full paper (reference) |
| `docs/28_11_25.pdf` | Project direction notes (Guillaume meeting) |
