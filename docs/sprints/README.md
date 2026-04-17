# MAPS — Sprint Index

| # | Sprint | Status | Branch | Est. effort |
|---|--------|--------|--------|------------:|
| 00 | [Remise en route (bootstrap)](sprint-00-remise-en-route.md) | 🟢 in progress | `refactor/clean-rerun` | 1 day |
| 01 | [Dedup & prune](sprint-01-dedup-and-prune.md) | ⚪ planned | `refactor/clean-rerun` | 1-2 days |
| 02 | [Extract MAPS core](sprint-02-extract-maps-core.md) | ⚪ planned | `refactor/extract-core` | 3 days |
| 03 | [Config-first migration](sprint-03-config-first.md) | ⚪ planned | `refactor/config-first` | 2-3 days |
| 04 | [Seed + logging + split monoliths](sprint-04-seeding-logging-split.md) | ⚪ planned | `refactor/monoliths` | 2-3 days |
| 05 | [Tests + CI](sprint-05-tests-and-ci.md) | ⚪ planned | `feat/tests-ci` | 1-2 days |
| 06 | Reproduction — Blindsight + AGL (local) | ⚪ planned | `repro/perceptual` | 2-3 days compute |
| 07 | Reproduction — SARL on Narval | ⚪ planned | `repro/sarl` | 1-2 weeks compute |
| 08 | Reproduction — MARL on Narval | ⚪ planned | `repro/marl` | 1-2 weeks compute |
| 09 | Docs + release | ⚪ planned | `feat/docs` | 2 days |

## Dependency graph

```
00 Bootstrap (security, pyproject, plan, hooks)
  │
  └─ 01 Dedup & prune  ← removes ~5k LOC dead/duplicated code
       │
       └─ 02 Extract MAPS core  ← src/maps/components/{second_order,cascade,losses}.py
            │
            ├─ 03 Config-first  ← config/maps.yaml + paths + typer CLIs
            │    │
            │    └─ 04 Seeding + logging + split monoliths
            │         │
            │         └─ 05 Tests + CI  ← pytest suite, GitHub Actions
            │              │
            │              ├─ 06 Reproduction perceptual (Mac-local)
            │              ├─ 07 Reproduction SARL (Narval)  ┐
            │              └─ 08 Reproduction MARL (Narval)  ├─ parallel
            │                                                ┘
            └─ 09 Docs + release
```

## Sprint conventions (inherited from PhiidResearch)

- **Branch:** `refactor/<sprint-scope>` or `feat/<sprint-scope>`
- **Sprint spec:** `docs/sprints/sprint-NN-kebab-name.md`
- **Report:** `docs/reports/sprint-NN-report.md` (written at end of sprint)
- **Commit prefix:** `feat(<scope>):`, `refactor(<scope>):`, `chore(<scope>):`, `fix(<scope>):`, `docs(<scope>):`
- **Definition of Done** is explicit at end of each sprint spec
- Sprints 06-08 are "reproduction" — no code changes, only runs + reports

## Non-goals (explicit)

- No Nature MI submission work (deferred, separate track)
- No new MAPS theoretical contributions
- No METTA-AI feature development (kept installable, not rebuilt)
- No port to PyTorch Lightning or any other training framework
