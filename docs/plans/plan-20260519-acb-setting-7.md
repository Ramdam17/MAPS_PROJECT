# Plan: ACB (Setting 7) port — SARL + MARL

**Date:** 2026-05-19
**Sprint:** Sprint 08 (closeout extension, parallel branch to plan-20260519-blindsight-agl-settings-4-5-6.md)
**Estimated complexity:** L (SARL: M, MARL: L)
**Author:** Rémy Ramadour + Claude

---

## Problem Statement

Paper Tables 6 and 7 each include a 7th row, **Setting 7 (ACB = Actor-Critic
Baseline, Young & Tian 2019, λ=0.8)**, applied to SARL (5 MinAtar games) and
MARL (4 MeltingPot substrates). The port currently raises `ValueError` on
`setting=7` (`src/maps/experiments/sarl/training_loop.py:885` —
`if setting not in _SETTING_TABLE: raise`) and the corresponding cells are
absent from every output directory. This is flagged in
`docs/reproduction/deviations.md` as `D-sarl-setting-7` (status ❌) with
phase action "E.1-E.5 — port ACB".

ACB is a **structurally different algorithm**, not a config switch on the
existing DQN/MAPPO code:

- Online actor-critic (policy + value heads) instead of DQN Q-network.
- Eligibility traces with `LAMBDA=0.8, GAMMA=0.99` (TD(λ) with replacing traces).
- RMSProp with initialization-debiasing update rule (custom — not
  `torch.optim.RMSprop`).
- No experience replay buffer.
- Entropy bonus with `BETA=0.01`.
- Activations `dSiLU = sigmoid(x)*(1+x*(1-sigmoid(x)))` and `SiLU = x*sigmoid(x)`.

Reference implementations vendored in
`external/paper_reference/sarl_ac_lambda.py` (498 lines, identical to
`external/MinAtar/examples/AC_lambda.py`). MARL ACB has **no vendored
student reference** — the paper applies AC_lambda to MeltingPot agents but
the multi-agent recipe (independent per-agent vs centralized critic with
trace decay) is not explicit; this needs clarification before MARL port.

**Out of scope:** the Know-Thyself Setting 4/5 work currently running on
Tamia (jobs 300277 + 300278, plan
`plan-20260519-blindsight-agl-settings-4-5-6.md`). The two efforts touch
disjoint trees (`src/maps/experiments/sarl/` and `marl/` vs
`blindsight/` and `agl/`) — they cannot conflict.

## Scientific Rationale

ACB is the paper's **negative-control baseline** — a competitive non-MAPS
algorithm trained on the same environments, to show that MAPS improvements
are not just "any actor-critic beats DQN". Paper Table 6:

- Breakout : ACB Train R = 12.36, Z = +7.49 vs Setting-1 (DQN baseline).
  *ACB is the strongest on this game — MAPS Setting 4 only reaches 6.06 Z.*
- Seaquest : ACB Train R = 0.81, Z = **-3.26** (significantly worse than DQN).
  *MAPS Setting 4 wins here at Z = 6.46.*

Reproducing both extremes is required to validate the headline claim that
MAPS' advantage is **environment-dependent, not algorithm-bias**. Skipping
Setting 7 would leave Tables 6 and 7 incomplete (28 of 35 SARL rows and 24 of
28 MARL rows).

**References:**
- Young & Tian (2019), *MinAtar: An Atari-inspired testbed for thorough and
  reproducible reinforcement learning experiments*, arXiv:1903.03176.
  AC(λ) algorithm at §5 (their Algorithm 1).
- Sutton & Barto (2018) §13.5 — actor-critic with eligibility traces.
- Paper TMLR submission §3.2 + Tables 6/7.
- Reference code: `external/paper_reference/sarl_ac_lambda.py` (vendored from
  `external/MinAtar/examples/AC_lambda.py`).

## Implementation Approach

### Two distinct sub-plans

```
Sub-plan A — SARL ACB (vendored reference exists)
   ├─ A.1 Port AC_lambda → src/maps/experiments/sarl/actor_critic.py
   ├─ A.2 Plumb Setting 7 dispatch in run_sarl.py
   ├─ A.3 Parity test vs external/paper_reference/sarl_ac_lambda.py
   ├─ A.4 sbatch array: 5 games × 3 seeds × Setting 7 = 15 cells
   └─ A.5 Aggregate + compare paper Table 6 row 7

Sub-plan B — MARL ACB (no vendored reference)
   ├─ B.1 Clarify multi-agent ACB recipe (per-agent independent? CTDE?)
   │      — open question for Rémy; possible sources: paper §3.2 text re-read,
   │        original MARL repo git log, email authors.
   ├─ B.2 Port to src/maps/experiments/marl/actor_critic.py
   ├─ B.3 Plumb Setting 7 in MarlSetting + scripts/run_marl.py
   ├─ B.4 Sanity check (no parity reference available — compare against
   │      paper Table 7 row 7 absolute numbers, accept ±2σ)
   ├─ B.5 sbatch array: 4 substrates × 3 seeds × Setting 7 = 12 cells
   └─ B.6 Aggregate + compare paper Table 7 row 7
```

Sub-plan A is **executable now** with a vendored ground truth. Sub-plan B
needs a methodology clarification first; recommendation: ship A, send to
Natalie with "MARL Setting 7 pending — recipe ambiguous in paper", then come
back to B with the actual answer.

### Data flow — SARL ACB

```
Input: config/training/sarl.yaml + --setting 7 + --game <g> + --seed <s>
  ↓ run_sarl.py: detect setting=7 → branch to ACBTrainer (not SarlTrainer)
  ↓ ACBTrainer.train(env, n_frames=500_000)
  ↓ ACNetwork(in_channels, num_actions) — Conv(16)+FC(128)+softmax(policy)+linear(value)
  ↓ Online step: (state, action, reward, next_state) → compute trace_potential,
                 entropy bonus, TD δ, update traces, RMSProp param update
Output: $SCRATCH/maps/outputs/sarl/<game>/setting-7/seed-<sd>/summary.json
        + losses_value.npy, returns.npy, policy.pt, value.pt
```

### Key components

| File | Change | Responsibility |
|---|---|---|
| `src/maps/experiments/sarl/actor_critic.py` (new) | `ACNetwork`, `ACBConfig`, `ACBTrainer` | Online AC(λ) with traces — port of vendored reference, no MAPS components |
| `src/maps/experiments/sarl/__init__.py` | export `ACBTrainer`, `ACBConfig` | API surface |
| `scripts/run_sarl.py` | `if setting == 7: trainer = ACBTrainer(...); else: SarlTrainer(...)` | Top-level dispatch |
| `src/maps/experiments/sarl/training_loop.py:874` | Extend `setting_to_config` to accept 7 → return a marker config (or raise a `SettingIsACB` exception caught by `run_sarl.py`) | Setting registry |
| `config/training/sarl_acb.yaml` (new) | `alpha: 0.00048828125`, `lambda: 0.8`, `gamma: 0.99`, `beta: 0.01`, `gamma_rms: 0.999`, `eps_rms: 0.0001`, `n_frames: 500000`, `validation_episodes: 10` | All hardcoded ACB hyperparams from paper Table 11 / vendored ref |
| `tests/parity/sarl/test_acb_parity.py` (new) | Tier-1 parity vs vendored reference: same seed → bitwise-equal network output after N=100 frames; loss curves match to 1e-5 | Guarantees the port is faithful |
| `tests/integration/sarl/test_acb_smoke.py` (new) | Each MinAtar env runs 200 frames, returns numerical and persists summary.json | Smoke before full sbatch |
| `scripts/slurm/sarl_setting_7.sh` (new) | array of 5 tasks (one per game), each loops 3 seeds. Time: 5-15h cascade (paper preamble) → request 24h | Production launch |
| `docs/reproduction/deviations.md` | Move `D-sarl-setting-7` to ✅ resolved, add row in B.7 audit table | Scientific record |

### Config snippet

```yaml
# config/training/sarl_acb.yaml — Setting 7 hyperparameters (Young & Tian 2019).
# All values from external/paper_reference/sarl_ac_lambda.py constants block
# (lines 31-39). Paper Table 11 does not list ACB hyperparams separately ;
# the vendored reference is the source of truth.

alpha: 0.00048828125              # AC(λ) step size (eq. 4 of YT19)
lambda: 0.8                       # trace decay
gamma: 0.99                       # discount
beta: 0.01                        # entropy bonus weight
gamma_rms: 0.999                  # RMSProp grad-sq EMA
eps_rms: 0.0001                   # RMSProp epsilon
min_denom: 0.0001                 # numerical floor in log(π) computation

# Training budget
n_frames: 500000                  # paper Table 11 "Number of frames"
validation_interval_frames: 10000 # eval every 10k env steps
validation_episodes: 10           # 10 eval episodes per checkpoint

# Logging
log_interval_episodes: 100        # match vendored ref printing cadence
```

## Risks & Unknowns

- **Risk A1 (HIGH): MinAtar import path on Tamia.** The vendored reference
  imports `from minatar import Environment`. Our SARL port uses `external/MinAtar`
  bundled. Need to confirm `minatar` is in the `sarl` extras group of
  `pyproject.toml` and that `.venv` resolves it offline (`UV_OFFLINE=1` in
  sbatch). → Mitigation: dry-run smoke before sbatch submission.

- **Risk A2: ACB algorithmic divergence under refactor.** The reference uses
  in-place tensor ops (`param.copy_(param + alpha*grad/...)`) without
  autograd tracking. The port must preserve this exactly — using
  `torch.optim.RMSprop` will NOT produce bit-identical updates because
  PyTorch's RMSprop omits the YT19 initialization-debiasing term
  `(1 - GAMMA_RMS**(t+1))` in the denominator. → Mitigation: port the
  custom update rule literally, write a parity test against the vendored
  reference's first N=100 steps.

- **Risk A3: Numerical floor differences.** `log(pi + MIN_DENOM)` with
  `MIN_DENOM=1e-4` is unconventional and changes gradient flow near
  zero-probability actions. → Mitigation: preserve the constant verbatim,
  document as deviation `D-sarl-acb-min-denom`.

- **Unknown B1 (HIGH, MARL-specific): per-agent vs centralized ACB.** Paper
  Table 7 Setting-7 has "2nd Net=No, Cascade=No" with `NA` in Dist Entropy
  for Harvest Partnership — suggesting ACB doesn't have the same entropy
  tracking as MAPS-MAPPO. Two possible recipes:
  - (a) N independent AC_lambda agents, one per substrate slot, no parameter
    sharing, no centralized critic.
  - (b) Shared-critic CTDE AC_lambda (Young & Tian don't describe this).
  → Will validate by: rereading paper §3.2 and §4 carefully ; if still
  ambiguous, ship sub-plan A first and ask the original authors.

- **Risk B2: ACB on MeltingPot RGB observations.** `ACNetwork` was designed
  for MinAtar's 10×10 binary channels. MeltingPot agents see 11×11 RGB
  (3 channels at higher resolution). Need conv re-sizing. → Mitigation:
  reuse the conv pattern from the existing MARL actor-critic
  (`src/maps/experiments/marl/`).

- **Risk B3: Compute budget.** Paper preamble says MAPS-MARL "~16h per seed
  on A100". ACB likely similar order, so 4 substrates × 3 seeds = 12 cells
  × ~16h = ~192 GPU-hours. → Mitigation: book Tamia H100 in chunks ;
  reuse current MARL sbatch infrastructure.

## Verification Plan

### Sub-plan A (SARL ACB)

- [ ] **Unit test** (`tests/unit/experiments/sarl/test_actor_critic.py` new):
      `ACNetwork(in_channels=4, num_actions=4)` forward shapes correct,
      softmax sums to 1 per row, value head is scalar per sample.
- [ ] **Parity test** (`tests/parity/sarl/test_acb_parity.py` new): vendored
      reference + port both run on identical (seed, env) for 100 frames →
      bitwise-equal `network.parameters()` and `traces` at the end (atol=1e-7).
- [ ] **Smoke test** (`tests/integration/sarl/test_acb_smoke.py` new): each
      of the 5 games runs 200 frames without crash, returns at least one
      validation point.
- [ ] **Numerical sanity**: Breakout 50k frames → mean validation return > 5
      (paper claims 12.36 at full 500k ; 1/10 budget should already separate
      from random ~1-2).
- [ ] **Paper Table 6 comparison** (per game, after full 500k frames × 3 seeds):
      compare `mean ± std` to paper row 7. Target: within ±2σ of paper.
- [ ] **No regression on Settings 1-6**: existing SARL parity tests still pass.

### Sub-plan B (MARL ACB)

- [ ] Methodology decision logged in `docs/reproduction/deviations.md` under
      new `D-marl-acb-recipe`.
- [ ] All sub-plan A verification items, applied to MARL substrates.
- [ ] Paper Table 7 row 7 within ±2σ for at least 3 of 4 substrates (entropy
      column is NA in paper, only Train R is compared).

## Definition of Done

### Sub-plan A — SARL Setting 7
- [ ] `src/maps/experiments/sarl/actor_critic.py` exists, exports
      `ACNetwork`, `ACBConfig`, `ACBTrainer`.
- [ ] `scripts/run_sarl.py` dispatches `--setting 7` to `ACBTrainer`.
- [ ] `config/training/sarl_acb.yaml` ships with paper-locked hyperparams.
- [ ] Tier-1 parity test green against vendored reference.
- [ ] 5 games × 3 seeds = 15 cells in `$SCRATCH/maps/outputs/sarl/<game>/setting-7/seed-<s>/`.
- [ ] `aggregate_sarl.py` recognises `setting-7` and emits the Table 6 row 7 comparison.
- [ ] `docs/reproduction/deviations.md` D-sarl-setting-7 → ✅ resolved.

### Sub-plan B — MARL Setting 7
- [ ] Methodology resolved (paper re-read or author confirmation).
- [ ] `src/maps/experiments/marl/actor_critic.py` ships.
- [ ] `MarlSetting` accepts `id="setting-7-acb"` with the agreed recipe.
- [ ] 4 substrates × 3 seeds = 12 cells in `$SCRATCH/maps/outputs/marl/<sub>/setting-7-acb/seed-<s>/`.
- [ ] Paper Table 7 row 7 within ±2σ on at least 3 substrates.
- [ ] `D-marl-setting-7` row added to `deviations.md` as ✅ resolved.

---

## Execution discipline

Per `feedback_microsprint_execution_discipline.md` and the parent Phase-α
plan: sub-plan A executes one micro-step at a time (A.1 → A.2 → A.3 → A.4
→ A.5), each with Orient→Do→Verify→Report→Commit→Wait. Sub-plan B starts
only after sub-plan A.5 (or in parallel if Rémy explicitly authorizes — but
the methodology clarification gate is non-negotiable).

Per `feedback_full_reproduction_commitment.md`: if A.3 parity fails, do not
ship A.4 with a less-faithful port. Fragment further (compare layer-by-layer,
gradient-by-gradient against vendored ref).

Per `feedback_paper_reproduction_preservation.md`: when the vendored
reference and the paper disagree on any hyperparam (e.g. `MIN_DENOM=1e-4`
is not in paper Table 11), the **paper wins** — but ACB itself isn't in
the paper's hyperparam tables, so the vendored reference IS the de-facto
source of truth here. Log every constant in the new
`config/training/sarl_acb.yaml` with provenance.

Parent plan (Know-Thyself Settings 4/5) continues independently — no
interaction.
