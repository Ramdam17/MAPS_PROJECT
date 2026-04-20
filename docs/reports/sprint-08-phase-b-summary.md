# Sprint-08 Phase B — closeout summary

**Période :** 2026-04-19.
**Branche :** `repro/paper-faithful`.
**Plan :** `docs/plans/plan-20260419-review-and-reproduce.md` §Phase B.
**Status :** ✅ closed — 13 sous-phases livrées atomiquement en 13 commits, 0 ligne de code touchée, ~2 000 lignes de documentation ajoutées.

## Objectif (rappel plan)

Produire la vérité écrite "papier ↔ monolithe student ↔ port courant" avant toute modification. Phase B est **audit-only** : aucune config, aucun code, aucun test n'est touché. Son delivrable gate Phase C (review composants), Phase D (review par experiment), Phase E (extensions ACB/MeltingPot/MARL port), Phase F (reproduction runs) et Phase G (rapport final).

## Livrables

### 4 nouveaux documents

| Doc | Contenu | Lignes | Rôle |
|---|---|---:|---|
| `docs/reproduction/paper_tables_extracted.md` | Tables 9, 10, 11, 12 papier (hyperparams) verbatim | 307 | Source de vérité hyperparams |
| `docs/reproduction/paper_equations_extracted.md` | Équations 1-18 papier verbatim (Know-Thyself, SARL, CL) | 369 | Source de vérité architecture |
| `docs/reproduction/paper_targets_extracted.md` | Tables 5, 6, 7 papier (z-scores cibles) verbatim | 280 | Source de vérité reproduction targets |
| `docs/reproduction/paper_vs_code_audit.md` | 5 audits croisés paper ↔ student ↔ port (SARL, SARL+CL, Blindsight, AGL, MARL) | 789 | Source de vérité divergences |

### 2 documents mis à jour

| Doc | Change | Rôle |
|---|---|---|
| `docs/reproduction/deviations.md` | +132 lignes : nouvelle section Sprint-08 avec 45 nouvelles entrées `D-*` | Checklist canonique divergences |
| `docs/reproduction/experiment_matrix.md` | Rewrite (+294 / −60) : N seeds corrigés, setting 7 ACB ajouté, 81 data rows target z-scores | Matrice de reproduction |

## 13 commits atomiques

```
fde3781  B.1   Table 11 (MinAtar hyperparams)             — 23 rows
140c1a2  B.2   Table 9 (Blindsight hyperparams)           — 11 rows
192d3aa  B.3   Table 10 (AGL hyperparams)                 — 13 rows
74bc27f  B.4   Table 12 (MeltingPot/MARL hyperparams)     — 13 rows + 2 flags
79f5345  B.5   Équations 1-18                              — 18 équations LaTeX
982b1be  B.6   Tables 5/6/7 (z-score targets)              — 81 data rows
13f5d18  B.7   Audit SARL                                  — 17 deviations, 7 🆘
676e6a8  B.8   Audit SARL+CL                               — 8 CL-specific + 17 hérités
9d27748  B.9   Audit Blindsight                            — 8 deviations, diag RG-002
4dc0332  B.10  Audit AGL                                   — 7 deviations, diag RG-003 structural
6e75642  B.11  Audit MARL                                  — 8 deviations + functional mapping
b1b75e0  B.12  Consolidation deviations.md                 — 45 IDs dans tableau unique
ef823a8  B.13  Rewrite experiment_matrix.md                — 7 settings, 81 target rows
```

## Findings principaux

1. **15 cas 🆘 "paper ≠ student"** — le code vendored dans `external/paper_reference/` **ne peut pas** avoir produit les z-scores Tables 5/6/7 du papier. Divergences hardcodées sur `GAMMA`, `lr_second_order`, `alpha_ema`, Adam betas, scheduler step_size, hidden_dim, etc. Implication Phase G : aligner sur paper Tables 9-12 produira des chiffres **différents** de ce que le student a publié.

2. **RG-002 (Blindsight) élucidé** : gap z=+0.40 mesuré vs z=9.01 papier causé par deux divergences port-vs-paper concomitantes :
   - `D-blindsight-hidden-dim` : config port = 100 (= input), paper = 60. Autoencoder sans bottleneck → comparison matrix ≈ 0 → wager aveugle.
   - `D-blindsight-metric-mismatch` : `evaluate()` reporte `discrimination_accuracy` = recall-only, paper "Main Task Acc" = overall accuracy.
   Fix Phase D.25 combine les deux hypothèses.

3. **RG-003 (AGL) élucidé — structural** : notre port n'a que `pre_train()`. Paper §AGL p.13 décrit un pipeline 3-phase (pre_train → training → testing). La phase 2 supervised training sur grammar A (post-pretrain, 2nd-order frozen, 12 epochs high / 3 epochs low) **n'est pas portée**. Fix Phase D.28 = porter `training()` depuis `external/paper_reference/agl_tmlr.py:904`.

4. **45 nouvelles déviations** tracées dans `deviations.md` (+ 4 originales + 3 G-IDs = **52 issues total**). Chaque entrée pointée vers la Phase D/E/F de fix.

5. **7 settings confirmés** (1-6 MAPS factorial + 7 ACB). Setting 7 (Actor-Critic Baseline, Young & Tian 2019, trace decay λ=0.8) absent de notre port actuel → à porter Phase E.1-E.5.

6. **D-marl-cascade-not-implemented — limitation paper-declared** : paper preamble Table 12 admet lui-même *"MAPS not implemented fully, only with simple 2nd order network with no cascade model due to limitations with computational resources"*. Settings MARL 2/4/5/6 ne sont **pas** paper-faithful cascade. À documenter honnêtement Phase G.

7. **Seeds corrigés** : paper = **500 Blindsight/AGL**, **3 SARL/SARL+CL/MARL** (pas N=10 comme nos docs disaient avant). Impact direct sur Phase F budget.

## Phase C readiness

Phase C = review composant-par-composant des 5 core MAPS modules (`cascade`, `second_order`, `losses`, `networks`, `utils`). Dépendances satisfaites :
- Équations source-of-truth disponibles (`paper_equations_extracted.md`).
- Divergences connues flagguées (via `deviations.md` + `paper_vs_code_audit.md`).
- Convention policy "paper wins" verrouillée Rémy 2026-04-19.

Ready pour démarrer C.1 (review `src/maps/components/cascade.py`).

## Changelog

- 2026-04-19 : Phase B closed. 13 sous-phases, 13 commits, 2 020 lignes doc, 0 ligne code.
