# Install Tamia (DRAC) — MAPS

**Cluster:** Tamia (DRAC) | **User:** rram17 | **Squelette:** P0.3 + P0.4 du Sprint 07 (2026-04-18).
Sera complété Phase 1 (P1.10).

## Allocation (P0.3)

- **Account SLURM :** `aip-gdumas85` (unique — **pas** de split `_cpu`/`_gpu` comme sur Narval).
- **QOS :** `normal`.
- `sshare -U -u rram17` :

  ```
  Account         User     RawShares  NormShares  RawUsage  EffectvUsage  FairShare
  aip-gdumas85    rram17           1    0.066667         0      0.000000   0.350379
  ```

- **Conséquence pour les sbatch :** `--account=aip-gdumas85` (sans suffixe). Toutes les occurrences de `aip-gdumas85_cpu` / `aip-gdumas85_gpu` dans le plan Sprint 07 doivent être remplacées.

## Partitions (P0.4, `sinfo -o "%P %G %l"`)

| Partition | GRES | MaxTime | Nodes |
|---|---|---|---|
| `cpubase_interac` | — | 6h | 8 |
| `cpubase_bynode_b1` / `bycore_b1` | — | 3h | 8 |
| `cpubase_bynode_b2` / `bycore_b2` | — | 12h | 6 |
| `cpubase_bynode_b3` / `bycore_b3` | — | **24h** | 4 |
| `gpubase_interac` | `gpu:h200:8` (2 nœuds), `gpu:h100:4` (8 nœuds) | 6h | 10 |
| `gpubase_bynode_b1` | `gpu:h200:8` (12), `gpu:h100:4` (53) | 3h | 65 |
| `gpubase_bynode_b2` | `gpu:h200:8` (11), `gpu:h100:4` (48) | 12h | 59 |
| `gpubase_bynode_b3` | `gpu:h200:8` (9), `gpu:h100:4` (40) | **24h** | 49 |

- **MIG : indisponible.** `GresTypes=gpu` uniquement, pas de slices MIG (`h100.*gb` absents). Les nœuds exposent le GPU entier : `gpu:h100:4` ou `gpu:h200:8`.
- **Impact plan :** mode `gpu_mig` retiré de Phase 1 P1.7 / Phase 2 P2.4-P2.6. Décision Phase 2 devient binaire : **CPU 4c** vs **H100 full** (ou H200 full, à benchmarker).
- **MaxTime global 24h** (tier b3). Pour 5M frames/cellule, si FPS > ~58/s → tient dans b3 sans chaining.

## Modules (à valider P0.5)

```bash
module load StdEnv/2023 python/3.12 cuda/12.6
```

## TODO (Phase 1, P1.10)

- [ ] Vérifier existence exacte `StdEnv/2023`/`cuda/12.6` sur Tamia (peut différer de Narval).
- [ ] Documenter `$SCRATCH` path + quotas (`diskusage_report`).
- [ ] Documenter install `uv` + `uv sync --extra sarl --extra dev`.
- [ ] Snippet sbatch minimal avec `--account=aip-gdumas85`, `--gres=gpu:h100:1`, modules, `--requeue`.
- [ ] Comparaison H100 vs H200 (après bench Phase 2).
