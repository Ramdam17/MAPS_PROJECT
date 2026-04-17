# MAPS_PROJECT — Cleanup TODO

## Technical Debt Register
_Last audited: 2026-04-17_
_Auditor: tech-debt agent_
_Scope: BLINDSIGHT, AGL, SARL, SARL_CL, MARL, METTA_

---

### Priority Matrix

| ID | File | Category | Severity | Effort | Description |
|----|------|----------|----------|--------|-------------|
| TD-001 | README.md:92-94 | Security | CRITICAL | Low | AWS access key + secret committed in plaintext |
| TD-002 | MARL/MAPPO-ATTENTIOAN/eval_meltingpot.sh:5; train_meltingpot.sh:5 | Security | CRITICAL | Low | wandb API token committed (`a2a1bab96ebb...`) |
| TD-003 | SARL/*.sh, SARL_CL/*.sh, MARL/meltingpot.sh | Reproducibility | HIGH | Low | Hardcoded `/home/juan-david-vargas-mazuera/...` and `/home/ubunto/...` absolute paths in 7 shell scripts |
| TD-004 | MARL/MAPPO-ATTENTIOAN/eval_meltingpot.sh:260,277; train_meltingpot.sh (same) | Reproducibility | HIGH | Low | Hardcoded `/home/zsheikhb/MARL/master{_skills}` PYTHONPATH and entrypoint paths from a second author |
| TD-005 | requirements.txt (416 lines) | Dependency | HIGH | Medium | Unpinned, unsorted, contains stdlib modules (`typing`, `dataclasses`, `contextvars`, `configparser`), duplicates, and non-PyPI names |
| TD-006 | BLINDSIGHT/Blindsight_TMLR.py (2419 L), AGL/AGL_TMLR.py (2785 L) | Maintainability | HIGH | High | Monolithic jupytext-exported files; no argparse/typer, hyperparameters hardcoded in `main()` |
| TD-007 | BLINDSIGHT/energy_tracker.py, AGL/energy_tracker.py, MARL/MAPPO-ATTENTIOAN/energy_tracker.py, SARL/MinAtar/examples/energy_tracker.py | Maintainability | HIGH | Low | Exact duplicate (all 563 lines, byte-identical) of `NvidiaEnergyTracker` across 4 locations |
| TD-008 | SARL/MinAtar/examples/maps_v1.py vs maps_v2.py | Maintainability | HIGH | Medium | Two near-identical MAPS DQN variants in same directory; maps_v2 adds only `count_parameters` and `curriculum` flag vs maps_v1 |
| TD-009 | SARL_CL/MinAtar/examples/maps.py vs SARL/MinAtar/examples/maps_v*.py | Maintainability | HIGH | High | MAPS components (CAE_loss, SecondOrderNetwork, DynamicLossWeighter, QNetwork, replay_buffer, DistillationLoss, target_wager, world_dynamics, train, evaluation, dqn) re-implemented, not imported |
| TD-010 | BLINDSIGHT/Blindsight_TMLR.py:91 + AGL/AGL_TMLR.py:89 | Maintainability | HIGH | Medium | `CAE_loss`, `FirstOrderNetwork`, `SecondOrderNetwork`, `compute_metrics`, `perform_linear_regression`, `plot_scaling`, `plot_violin`, `load_*_from_csv`, `run_setting_experiment` duplicated between BLINDSIGHT and AGL |
| TD-011 | BLINDSIGHT/Blindsight_TMLR.py, AGL/AGL_TMLR.py, SARL/*/maps_v*.py | Reproducibility | CRITICAL | Low | No `torch.manual_seed`, `np.random.seed`, or `random.seed` calls anywhere in the 4 core experiment scripts despite a `seeds` loop variable — results are not reproducible |
| TD-012 | MARL/MAPPO-ATTENTIOAN/dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl | Dependency | HIGH | Low | Linux-only binary wheel committed (~tens of MB); bloats repo and unusable on macOS dev machines |
| TD-013 | METTA/metta_MAPS/awscliv2.zip | Hygiene | HIGH | Low | AWS CLI installer binary committed |
| TD-014 | repo root: no .gitignore | Hygiene | HIGH | Low | No top-level .gitignore; `__pycache__/*.pyc` committed (BLINDSIGHT, AGL, MARL), plus `METTA/metta_MAPS/wandb/run-*` directories with metadata and logs |
| TD-015 | AGL/energy_data/*.csv,png (30 files), BLINDSIGHT/energy_data/*.csv,png (24 files) | Hygiene | MED | Low | Experimental run artifacts from 2025-05-22/23 committed to repo |
| TD-016 | MARL/MAPPO-ATTENTIOAN/onpolicy/algorithms/happo/happo_trainer(1).py | Dead code | MED | Low | Filename with `(1)` indicates an accidental download-duplicate next to `happo_trainer.py` |
| TD-017 | MARL/MAPPO-ATTENTIOAN/onpolicy/envs/mpe/environment_old.py, onpolicy/algorithms/utils/cnn_original.py | Dead code | MED | Low | `_old` / `_original` shadow files kept alongside canonical versions |
| TD-018 | SARL/MinAtar/build/, SARL_CL/MinAtar/build/ | Hygiene | MED | Low | Setuptools `build/lib/...` trees checked in (full duplicate of `minatar/` package) |
| TD-019 | SARL/MinAtar/minatar/environment.py:13, gym.py:9; SARL_CL mirror | Correctness | MED | Low | Bare `except:` swallowing any error on optional GUI/gym import |
| TD-020 | AGL/AGL_TMLR.py:2502-2504; BLINDSIGHT/Blindsight_TMLR.py (`load_*_from_csv`) | Correctness | MED | Low | `except Exception as e: print(...); return None` — caller can't distinguish missing file from parse error; no logging |
| TD-021 | METTA/metta_MAPS/mettagrid/mettagrid/util/actions.py:351,364,376; metta/agent/*; devops/aws/batch/*.py | Correctness | MED | Medium | ~20 `except Exception: pass` sites swallowing errors silently (see grep in audit) |
| TD-022 | BLINDSIGHT/Blindsight_TMLR.py, AGL/AGL_TMLR.py, SARL/*/maps_v*.py | Logging | HIGH | Medium | Zero use of `logging`; everything uses `print()` (315 print calls across these 4 files). No log level, no timestamps, no file handler |
| TD-023 | BLINDSIGHT/Blindsight_TMLR.py:1677 | Correctness | MED | Low | Function `plot_scaling_discrimination` defined twice (line 1521 and 1677); second definition silently shadows first |
| TD-024 | AGL/AGL_TMLR.py:26, BLINDSIGHT/Blindsight_TMLR.py:26,35 | Maintainability | LOW | Low | `import torch_optimizer as optim2` imported twice; `#!pip install ...` shell magics left in converted .py file |
| TD-025 | AGL/AGL_TMLR.py, BLINDSIGHT/Blindsight_TMLR.py | Correctness | MED | Medium | No explicit device management (`torch.device`, `.to(device)`); relies on default, will silently run on CPU on non-CUDA hosts |
| TD-026 | MARL/MAPPO-ATTENTIOAN/onpolicy/envs/env_wrappers.py:17 and 5 methods (66,78,92,99,175,179,188) | Maintainability | LOW | Low | Abstract-like methods with bare `pass` and no `raise NotImplementedError` / ABC decorator |
| TD-027 | repo root | Test | HIGH | High | No `tests/` for BLINDSIGHT, AGL, SARL, SARL_CL, MARL. Only METTA vendors its own tests. No pytest config, no CI workflow |
| TD-028 | repo root | CI | HIGH | Medium | No `.github/workflows`, no pre-commit, no ruff/black config. `pyproject.toml` absent at top level |
| TD-029 | METTA/metta_MAPS/wandb/run-20250629_204434-my_experiment7/ | Hygiene | MED | Low | A full wandb run directory (metadata, output.log, config.yaml) committed inside METTA |
| TD-030 | AGL/AGL_TMLR.py:2164 (`main`), BLINDSIGHT/Blindsight_TMLR.py:2506 (`main`) | Maintainability | HIGH | High | `main()` ~350 lines with hard-coded hyperparameter lists (`hidden_sizes = [30,40,...]`); no CLI, only `Training = True` flag to toggle branches |
| TD-031 | SARL/MinAtar/examples/maps_v2.py:23, maps_v1.py:23 | Documentation | LOW | Low | Stale usage example in header: `python examples/dqn_2nd_order.py ...` but file is `maps_v2.py` |
| TD-032 | requirements.txt:1-10 | Dependency | HIGH | Low | Lists Python stdlib modules as pip deps: `typing`, `dataclasses`, `contextvars`, `configparser`, `contextlib2` — installing these from PyPI will shadow stdlib and break things on 3.12 |
| TD-033 | README.md:87-94 | Documentation | MED | Low | After rotating the leaked key, the README still instructs users to `export AWS_ACCESS_KEY_ID=...` with no explanation of which bucket / which IAM policy is needed |
| TD-034 | BLINDSIGHT/Blindsight_TMLR.py, AGL/AGL_TMLR.py | Reproducibility | MED | Low | `initialize_global()` mutates module-level globals instead of returning state — inter-run bleed risk when called from notebooks |
| TD-035 | SARL/SARL_Training_Optimized.sh:22 (`/home/ubunto/...`) + SARL/SARL_Training_Standard.sh:22 | Reproducibility | MED | Low | Typo `ubunto` (not `ubuntu`) suggests copy-paste from a second author's machine; indicates the scripts were never meant to run off-box |
| TD-036 | METTA/metta_MAPS/wandb/run-*/files/wandb-metadata.json | Reproducibility | LOW | Low | Committed wandb metadata reveals internal paths `/home/ubunto/MSc_CS/MAPS_PROJECT/...` — confirms two conflicting authors |
| TD-037 | MARL/MAPPO-ATTENTIOAN/{eval,train}_meltingpot.sh | Reproducibility | MED | Low | `conda init bash` + `conda shell.bash activate marl` + `conda activate marl` — triple activation, relies on a `marl` env that is not documented anywhere |
| TD-038 | MARL/MAPPO-ATTENTIOAN/onpolicy/algorithms/utils/utilities/ (12 files) | Maintainability | MED | Medium | Vendored "RIMs / SCOFF / set_transformer" code with no attribution comments or README; unclear what is hand-modified vs upstream |
| TD-039 | METTA/metta_MAPS/ | Dependency | MED | High | A full fork of the `metta` monorepo (mettagrid, mettascope, devops, tf, observatory) vendored rather than pinned as a git submodule — thousands of files maintained by an external team |
| TD-040 | SARL/SARL_Plot_Results.sh:15 | Reproducibility | HIGH | Low | References `/home/juan-david-vargas-mazuera/ICML-RUNS/WorkshopPaper/REPO/MinAtar/results/` — plotting will fail immediately outside original machine |

---

### Debt Details

#### TD-001 — Leaked AWS credentials
**File:** README.md:92-94
**Severity:** CRITICAL — these keys are in git history and must be considered compromised.
**Fix:** (1) Rotate the IAM user NOW. (2) Remove lines. (3) Rewrite git history with `git filter-repo` or accept leak and revoke. (4) Replace with `aws configure` instructions.
**Effort:** 30 min (plus coordination with whoever owns the AWS account).

#### TD-002 — Leaked wandb token
**File:** MARL/MAPPO-ATTENTIOAN/eval_meltingpot.sh:5, train_meltingpot.sh:5
**Fix:** revoke token in wandb settings; use `WANDB_API_KEY` env var read from a `.env` file that is gitignored.
**Effort:** 15 min.

#### TD-011 — No seed control in MAPS experiments
**Evidence:** `grep -n "manual_seed\|np\.random\.seed\|random\.seed"` in BLINDSIGHT/Blindsight_TMLR.py and AGL/AGL_TMLR.py returns 0 hits. The word "seed" is used only as a *loop variable name* (e.g. `for seed in range(seeds)`). SARL maps_v1/v2 accept `--seed` via argparse but never call `torch.manual_seed(args.seed)`.
**Impact:** every run produces different results; the "replication across seeds" claim in the paper is not achievable from this code without re-instrumentation.
**Fix:** central `set_seed(seed)` helper that seeds `random`, `numpy`, `torch`, `torch.cuda`, and sets `torch.backends.cudnn.deterministic = True`.
**Effort:** 1 h to add + verify.

#### TD-007 / TD-008 / TD-009 / TD-010 — Copy-paste of MAPS core
Four separate copies of the MAPS components:
- `BLINDSIGHT/Blindsight_TMLR.py` (2419 L)
- `AGL/AGL_TMLR.py` (2785 L)
- `SARL/MinAtar/examples/maps_v1.py` / `maps_v2.py`
- `SARL_CL/MinAtar/examples/maps.py`

Shared re-implemented symbols include: `CAE_loss`, `FirstOrderNetwork`, `SecondOrderNetwork`, `DynamicLossWeighter`, `count_parameters`, `compute_metrics`, `perform_linear_regression`, `plot_scaling`, `plot_violin*`, `initialize_tracker`, `stop_get_results_tracker`, and the entire `energy_tracker.py` module (byte-identical 563-line copy in 4 dirs).
**Fix (proposed, needs approval):** extract a `maps_core/` package with `networks.py`, `losses.py`, `training.py`, `energy.py`, then import from each experiment.
**Effort:** High (1–2 days) but eliminates ~6 000 lines of duplicate code.

#### TD-022 — Print-based logging
`print(` occurs 315 times across the 4 core experiment files. No `logging` import. This violates the project convention ("Log with Python `logging` module (not `print`) in production code"). Side-effect: impossible to control verbosity, no timestamps in long training runs.

#### TD-023 — Duplicate function definition shadowing
BLINDSIGHT/Blindsight_TMLR.py defines `plot_scaling_discrimination` twice (lines 1521 and 1677). Python silently keeps the second. Whatever the first version did is dead code.

#### TD-032 — stdlib modules in requirements.txt
Lines include `typing`, `dataclasses`, `contextvars`, `configparser`, `contextlib2`, `backports.tarfile`. Installing the PyPI `typing` backport on Python 3.12 is known to shadow the stdlib and cause `AttributeError: module 'typing' has no attribute 'ParamSpec'`. This alone can break a clean install.

---

### Not-yet-investigated / flagged for follow-up
- METTA/metta_MAPS vendoring (TD-039): decide whether to replace with a git submodule at a pinned commit.
- MARL RIM/SCOFF utilities (TD-038): locate upstream and restore attribution per their license.
- SARL/MinAtar `build/` trees (TD-018): confirm they can be regenerated from `setup.py` before deletion.

---

### Summary
- CRITICAL: 3 items (TD-001, TD-002, TD-011)
- HIGH: 16 items
- MEDIUM: 17 items
- LOW: 4 items
- **Total estimated remediation effort:** ~5–7 engineer-days for HIGH/CRITICAL; ~2 weeks for full register including the MAPS-core refactor (TD-007..TD-010) and test/CI bootstrapping (TD-027, TD-028).

### Recommended order of attack
1. **Day 1 (security):** TD-001, TD-002, TD-033 — rotate keys, purge tokens, rewrite README.
2. **Day 1 (repo hygiene):** TD-012, TD-013, TD-014, TD-015, TD-018, TD-029 — add top-level `.gitignore`, remove binaries/artifacts, one commit.
3. **Day 2 (reproducibility minimum):** TD-003, TD-004, TD-040, TD-011 — parametrize shell scripts, add `set_seed()`.
4. **Day 3 (deps):** TD-005, TD-032 — migrate to `pyproject.toml` + `uv.lock`, drop stdlib entries, pin versions.
5. **Week 2 (refactor):** TD-006, TD-007..TD-010, TD-022, TD-030 — extract `maps_core/`, add CLI, switch to `logging`.
6. **Week 2 (safety net):** TD-027, TD-028 — pytest scaffold + ruff/black + GitHub Actions.
