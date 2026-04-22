# MARL run outputs — handover for analysis

**Context :** this document hands over the MARL reproduction run (Phase E,
sprint 09) to whoever will aggregate the per-cell outputs into paper Table 7
and any downstream statistical analysis.

**Produced by :** `scripts/run_marl.py` driven by `scripts/slurm/marl_array.sh`
(SLURM job 256900 on DRAC Tamia, run 2026-04-21 → ~2026-04-25).

---

## 1. Directory layout

Each training cell writes one directory under a fixed root. The root
differs depending on where you're reading from :

| location | root path |
|---|---|
| **DRAC Tamia compute / login node** (during + just after the run) | `$SCRATCH/maps/outputs/marl/` |
| **Project-local** (post-rsync, what ends up in the git-tracked tree) | `outputs/marl/` |

The extra `maps/` level on `$SCRATCH` is intentional — it namespaces MARL
outputs away from the other MAPS domains (`sarl`, `blindsight`, `agl`,
`sarl_cl`) that also write under `$SCRATCH/maps/outputs/`. Once rsync'd
into the repo, `outputs/` is the project root and the `maps/` level
disappears.

**Per-cell structure** (identical in both roots) :

```
<root>/
└── <substrate>/
    └── setting-<setting_id>/
        └── seed-<N>/
            ├── metrics.json        ← the file you read
            └── checkpoint.pt       ← resume-only, ~160 MiB, can be ignored for analysis
```

**Substrates** (4) :
- `commons_harvest_closed`        — 6 agents (paper §A.4)
- `commons_harvest_partnership`   — 4 agents
- `chemistry`                     — 8 agents (substrate name
  `chemistry__three_metabolic_cycles_with_plentiful_distractors`)
- `territory_inside_out`          — 5 agents

**Settings** (6, cf. `config/experiments/factorial_marl.yaml`) :
- `baseline`              — Setting-1 in paper (no meta, no cascade)
- `cascade_1st_no_meta`   — Setting-2 (cascade on 1st-order only)
- `meta_no_cascade`       — Setting-3 (meta only)
- `maps`                  — Setting-4 (paper's "MAPS", meta + cascade 1st-order)
- `meta_cascade_2nd`      — Setting-5 (meta + cascade on wager/2nd-order)
- `meta_cascade_both`     — Setting-6 (full, meta + cascade on both)

**Seeds** (3) : `42`, `43`, `44`.

→ **Total : 4 × 6 × 3 = 72 cells.**

---

## 2. `metrics.json` schema

```json
{
  "meta": {
    "substrate": "commons_harvest_closed",
    "setting": {
      "id": "baseline",
      "label": "Baseline MAPPO (no meta, no cascade)",
      "meta": false,
      "cascade_iterations1": 1,
      "cascade_iterations2": 1
    },
    "seed": 42,
    "num_env_steps": 300000,       // paper §4 p.15
    "episode_length": 1000,
    "n_rollout_threads": 1,
    "num_agents": 6,
    "elapsed_s": 5398.3            // wall-time for this cell
  },

  "episodes": [
    // 300 entries : episode_length × N_rollout_threads = 1000, 300k / 1000 = 300 eps
    {
      "episode": 0,
      "total_steps": 1000,
      "episode_return_mean":      27.33,                        // ── Paper Table 7 "Training Rewards"
      "episode_return_per_agent": [75.0, 89.0, 0.0, 0.0, 0.0, 0.0],
      "per_agent": [                                            // list of N_agents dicts
        {
          "value_loss":       1.002,
          "policy_loss":     -0.0038,
          "dist_entropy":     2.079,                            // ── Paper Table 7 "Dist Entropy Value"
          "actor_grad_norm":  0.267,
          "critic_grad_norm": 24.164,
          "ratio":            0.999,                            // PPO importance-sampling ratio (sanity)
          "wager_loss_actor":  0.0,                             // 0.0 when setting.meta == false
          "wager_loss_critic": 0.0
        },
        // ... (num_agents - 1) more agents
      ]
    },
    // ... 299 more episodes
  ]
}
```

**Size per cell :** ≈ 750 KiB. Full 72-cell tree ≈ 55 MiB — fits anywhere.

---

## 3. Mapping to paper Table 7

Paper Table 7 (p.16, "Training and validation rewards, Z-score, and significant
results for MARL") reports two quantities per (substrate × setting) cell :

| Paper column | Source field | Aggregation recipe |
|---|---|---|
| **Training Rewards** (e.g. 19.52 ± 0.71) | `episodes[*].episode_return_mean` | mean ± std of the **last 50 episodes**, then mean ± std across the 3 seeds |
| **Dist Entropy Value** (e.g. 1.89 ± 0.01) | `episodes[*].per_agent[*].dist_entropy` | per episode : mean across agents ; then mean ± std of the last 50 eps ; then across seeds |
| **Z-score** | — | `(setting_mean - baseline_mean) / baseline_std`, computed per substrate ; baseline = Setting-1 of that substrate |

**"Last 50 episodes"** matches the paper's convergence-window convention and
is what the E.17b3 3-seed validation used (1.4σ_paper match on
`commons_harvest_closed` × `baseline`).

---

## 4. Starter aggregation script

Save as `scripts/aggregate_marl_table7.py` (not yet written — this is the
recipe) :

```python
"""Reproduce paper Table 7 from the MARL run outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT_ROOT = Path("outputs/marl")           # adjust if reading from $SCRATCH directly
LAST_N   = 50                             # convergence window

rows = []
for metrics_path in OUT_ROOT.rglob("metrics.json"):
    d = json.load(open(metrics_path))

    # Skip historical / validation runs that don't belong in the factorial.
    seed_dir = metrics_path.parent.name   # e.g. "seed-42" ; skip "seed-42-validate", "seed-42-bench1M"
    if not seed_dir.replace("seed-", "").isdigit():
        continue

    # Skip pre-E.17b2 runs that lack the reward field (safety net).
    if "episode_return_mean" not in d["episodes"][0]:
        continue

    last = d["episodes"][-LAST_N:]
    rewards   = np.array([ep["episode_return_mean"] for ep in last])
    entropies = np.array([
        np.mean([a["dist_entropy"] for a in ep["per_agent"]])
        for ep in last
    ])

    rows.append({
        "substrate":    d["meta"]["substrate"],
        "setting":      d["meta"]["setting"]["id"],
        "seed":         d["meta"]["seed"],
        "reward_mean":  rewards.mean(),
        "reward_std":   rewards.std(ddof=1),
        "entropy_mean": entropies.mean(),
        "entropy_std":  entropies.std(ddof=1),
    })

df = pd.DataFrame(rows)

# Cross-seed aggregation (this is what the paper actually reports).
agg = df.groupby(["substrate", "setting"]).agg(
    training_reward_mean=("reward_mean", "mean"),
    training_reward_std =("reward_mean", "std"),   # std across seeds
    entropy_mean        =("entropy_mean", "mean"),
    entropy_std         =("entropy_mean", "std"),
    n_seeds             =("seed", "count"),
).reset_index()

# Z-score against per-substrate baseline.
baseline = agg[agg.setting == "baseline"].set_index("substrate")
agg["reward_zscore"] = agg.apply(
    lambda r: (r.training_reward_mean - baseline.loc[r.substrate, "training_reward_mean"])
              / max(baseline.loc[r.substrate, "training_reward_std"], 1e-9)
              if r.setting != "baseline" else np.nan,
    axis=1,
)

print(agg.to_string(index=False))
agg.to_csv("outputs/marl/table7_reproduction.csv", index=False)
```

---

## 5. Things to keep in mind

**What to include in the factorial aggregation :**
- `outputs/marl/<substrate>/setting-<id>/seed-{42,43,44}/metrics.json` — **yes** (72 cells).

**What to exclude :**
- `seed-*-validate/` — the E.17b3 3-seed gate runs on `commons_harvest_closed × baseline` only (reference history).
- `seed-*-bench1M/` — the E.17b 1 M-step sanity-check run on
  `commons_harvest_closed × baseline × seed 42` (also predates reward logging
  so `episode_return_mean` is absent from its JSON ; the starter script
  above skips it automatically).

**Known port deviations that matter for interpretation** :
- `num_env_steps = 300000` per paper §4 text, not 15 M (Table 12 appears to
  be a template ; same table has `Critic lr = 100` which is clearly a typo).
  See `config/training/marl.yaml` header for the full justification.
- EMA wager α = 0.45 (paper eq.13), not 0.25 (student code bug).
- Wager condition : `r_t > EMA_t` (paper eq.14), not `EMA_t > 0` (student).
- Scope : RIM / SCOFF / PopArt / MLPBase omitted per E.5 scope lock — we
  implement the plain MAPPO + MAPS path only.

Full list and rationale : `docs/reproduction/deviations.md`.

**Stochasticity** : 3 seeds is the same N as the paper. Expect Z-scores
comparable to the paper at 1-2σ tolerance (E.17b3 validation showed
commons_harvest_closed baseline = 20.53 ± 1.37 vs paper 19.52 ± 0.71 →
1.4σ_paper). Our cross-seed std tends to be ~1.5-2× paper's — likely due
to `n_rollout_threads = 1` vs whatever vectorisation the paper used. This
widens error bars but preserves mean separability ; z-scores will be
slightly more conservative than the paper's.

---

## 6. Provenance & reproducibility

- Git branch at run time : `repro/paper-faithful` (commits up to `0e0d666`).
- Python 3.11.4 (`/cvmfs/.../python/3.11.5` module + `.venv-marl`).
- PyTorch 2.5.1+cu121 ; CUDA driver 12.8 ; H100 80 GB.
- Each cell's exact commit, hyperparameters, and RNG seeds are in its
  `metrics.json["meta"]` + the repo git log. The `checkpoint.pt` file
  contains the full model + optimizer + RNG state if bit-exact replay is
  needed.
