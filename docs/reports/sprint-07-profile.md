# Sprint 07 Phase 2 — Bench + Profile (in progress)

**Status:** overnight batch submitted 2026-04-19 00:22 UTC.  
**Plan ref:** `docs/plans/plan-20260418-sprint07-sarl-tamia.md` §Phase 2.

## Key finding surfaced during P2.1

**Setting 6 (full MAPS) is ≈ 40 × slower than setting 1 on CPU.** The Phase 0 smoke (P0.9) measured setting 1 (vanilla DQN) at ~294 fps (10 k frames in 34 s). The Phase 2 smoke P2.1 on setting 6 (cascade both networks, 50 iter) showed ~**7 fps** CPU — consistent with cascade × 50 forward passes dominating update cost. This changes Phase 4 sizing: setting 6 at 5 M frames on CPU = ~8 days/cell, infeasible without GPU acceleration.

Consequence: the Phase 2 bench grid now covers **both ends** (setting 1 fast baseline, setting 6 worst case) on CPU and GPU so the CPU-vs-GPU decision (P2.6) can be setting-aware.

## Overnight job map (2026-04-19 ~00:22 UTC)

| Job ID | What | Setting | Device | Frames | --time | Writes |
|---|---|---|---|---|---|---|
| 235781 | bench A | 1 | cpu_4c | 500 k | 01:00:00 | `outputs/bench/bench-cpu_4c-s1-seed42-<rev>.json` |
| 235782 | bench B | 6 | cpu_4c | 50 k | 03:00:00 | `outputs/bench/bench-cpu_4c-s6-seed42-<rev>.json` |
| 235783 | bench C | 1 | gpu_full (h100:4) | 500 k | 02:00:00 | `outputs/bench/bench-gpu_full-s1-seed42-<rev>.json` |
| 235784 | bench D | 6 | gpu_full (h100:4) | 50 k | 03:00:00 | `outputs/bench/bench-gpu_full-s6-seed42-<rev>.json` |
| 235785 | profile E | 6 | cpu | 25 k | 03:00:00 | `outputs/bench/prof-6-42-<rev>.{out,txt}` |
| 235786 | profile F | 1 | cpu | 100 k | 01:00:00 | `outputs/bench/prof-1-42-<rev>.{out,txt}` |

Filename schema is `bench-<mode>-s<SETTING>-seed<SEED>-<rev>.json` so multi-setting benches don't collide (fixed before job launch).

## (to be filled in) Bench results

| mode | setting | frames | wall_s | fps | updates/s | peak_vram_mb | peak_rss_mb |
|---|---|---|---|---|---|---|---|
| cpu_4c | 1 | 500 k | — | — | — | 0 | — |
| cpu_4c | 6 | 50 k | — | — | — | 0 | — |
| gpu_full | 1 | 500 k | — | — | — | — | — |
| gpu_full | 6 | 50 k | — | — | — | — | — |

## (to be filled in) Decision — partition for Phase 4

- [ ] Setting 1 extrapolation (5 M frames × fps_5 M): CPU ≈ TBD h, GPU ≈ TBD h
- [ ] Setting 6 extrapolation: CPU ≈ TBD h, GPU ≈ TBD h
- [ ] Ratio GPU/CPU per setting
- [ ] Per-setting partition verdict
- [ ] Updated `sarl_array.sh --time` / `--array=0-299%N`

## (to be filled in) Profile top-hotspots

### Setting 1 (baseline)

(paste `outputs/bench/prof-1-42-<rev>.txt` top 20)

### Setting 6 (cascade full MAPS)

(paste `outputs/bench/prof-6-42-<rev>.txt` top 20)

### Crosswalk vs Phase 3 optim hypotheses

| # | Piste | Measured cost share | Verdict |
|---|---|---|---|
| O-1 | `target_wager` (sarl/data.py:93-118) — boucle Python | TBD | TBD |
| O-2 | `non_terminal_idx` (trainer.py:167) list-comp | TBD | TBD |
| O-3 | `get_state` (data.py:82-90) alloc + permute + to | TBD | TBD |

## Known gaps / quirks

- `bench_sarl.py` filename not setting-aware — issue #TODO.
- GPU on H100 is **whole-node** (4 GPUs); single-GPU benches are wasteful but
  unavoidable on Tamia. Future sprint could pack N cells per node with
  CUDA_VISIBLE_DEVICES partitioning.
- `235767` (old CPU smoke setting 6 50k) left running until it hits its
  01:00:00 time limit — setting 6 on CPU only reaches ~15 k frames in that
  budget; metrics.json will NOT be written (persist is end-of-run).
  Ignored, not re-submitted.
