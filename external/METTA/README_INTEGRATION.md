# METTA — vendored integration

This is a vendored copy of the METTA-AI project (cognitive evals substrate), used
as an exploratory extension of MAPS. It lives in `external/` because it manages
its own environment and must not be mixed with the root `maps` package.

## Environment isolation

METTA has its own `pyproject.toml` + `uv.lock` + `CLAUDE.md` + `AGENTS.md`. Do
**not** install METTA's deps into the root `maps` env — Ray / PufferLib /
mettagrid versions diverge.

```bash
cd external/METTA
uv sync          # METTA's own environment
uv run pytest    # METTA's own tests
```

## Relationship to the parent repo

- Root `maps` package: Blindsight, AGL, SARL, SARL+CL, MARL domains (Sprint 02+)
- `external/METTA`: independent exploratory substrate, not required to reproduce
  the TMLR paper results

When working inside `external/METTA/`, follow its own `CLAUDE.md` (its style
guide, conventions, and build system are authoritative there).

## Why vendored instead of a submodule

Per Sprint 00 discussion: the upstream METTA repo is evolving fast and we want
to freeze a known-good snapshot while we finish the MAPS reproduction. Will
revisit after Sprint 09.
