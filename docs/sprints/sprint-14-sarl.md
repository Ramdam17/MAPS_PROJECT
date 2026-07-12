# Sprint 14 — SARL rewrite + tests + paper reproduction

**Status:** 🔵 open (2026-07-12)
**Branch:** `refactor/sarl` (branchée depuis `6f07916` — cœur partagé complet)
**Owner:** Rémy Ramadour
**Depends on:** Sprint 11 ✅ (core/) + Sprint 12 ✅ (Blindsight) + Sprint 13 ✅ (AGL, le patron trainer/pool/cli)
**Numerical reference:** `external/paper_reference/sarl/maps_v1.py` (v1 **paper-canonical**,
produit les chiffres paper Table 6 ; lancé par `SARL_Training_Standard.sh` à 2M frames, α=0.25).

---

## Context

SARL = MinAtar DQN + briques MAPS (2×2 factorial → 6 settings, Table 6). Premier domaine
**RL** du rebuild — bien plus lourd que Blindsight/AGL : boucle d'entraînement sur ~2M
frames, replay buffer, ε-greedy, cascade dans la sélection d'action ET dans l'update.

Réutilise `core/` (cascade, second_order, losses) + `utils/`. Ajoute `domains/sarl/`.
Ne réutilise PAS `networks/first_order_mlp` (SARL a ses propres réseaux conv).

## Goal

SARL settings 1-6 **paper-faithful** (bugs compris) vs `maps_v1.py`, parité adaptée au RL.
Reproduction fidèle, pas de correction.

---

## Décisions Day-1 (verrouillées 2026-07-12)

### D14.1 — Source & variante : v1 uniquement
Reproduire `external/paper_reference/sarl/maps_v1.py` (v1, canonique Table 6). **PAS** v2
(`maps_v2.py`, la variante erronée qui a causé le post-mortem Sprint-09
`docs/reproduction/sarl-postmortem-20260524.md`). La branche ne contient QUE le code v1.
v1 diffère de v2 sur 4 pièces structurelles : Q-head 1024 (v1) vs 128 (v2) ; décodeur dédié
`Linear(128,1024)` (v1) vs tied-weight (v2) ; `comparison_layer` 1024×1024 actif (v1) vs
commenté (v2) ; cascade sur Output 1024-d (v1) vs Hidden 128-d (v2). Budget : 2M frames +
α=0.25 (v1) vs 500k + α=0.45 (v2).

### D14.2 — Baseline fidèle (bugs compris, comme D13.2)
Reproduire maps_v1.py tel quel, bugs de l'audit inclus (env non seedé, scheduler step_size=1,
comportement cascade sur l'archi v1). **Ne PAS** porter F1 (RNG resume) ni F5 (chemins aggregate).

### D14.3 — Env MinAtar non seedé (fidèle)
La source ne seede pas l'env MinAtar (audit T-1). Fidèle = non seedé, pour la phase parité.
Le seeding (règle projet + décision Guillaume) devient une option/décision **production**
séparée, non mélangée à la reproduction.

### D14.4 — Parité adaptée au RL (pas de bit-exact run-complet)
- Tier 1 — wrapper env MinAtar + replay buffer (formes, indexation, transitions).
- Tier 2 — forward réseau v1 bit-exact (les 4 pièces structurelles v1) + cascade.
- Tier 3 — un pas d'update DQN bit-exact (cible Q détachée, terminal mask, loss, backward, step).
- Tier 4-light — court rollout + quelques updates : déterminisme (même seed → mêmes poids) +
  parité d'orchestration inline (two-loss + ordre des opés).

### D14.5 — Périmètre ; ACB reporté
Modules : `data` (replay buffer + wrapper MinAtar), `model_v1` (SarlQNetworkV1 +
SarlSecondOrderNetworkV1), `rollout` (ε-greedy + sélection cascade), `trainer` (update DQN +
two-loss), `training_loop` (boucle RL + checkpoint), `evaluate`, `cli`.
**ACB (setting 7, `sarl_ac_lambda.py`) reporté** — algorithme séparé hors factoriel 6-cellules ;
fera une phase 14.x finale ou un Sprint 14b après les settings 1-6.

---

## Plan de phases

| Phase | Contenu | Commit |
|-------|---------|--------|
| 14.A | ouvrir sprint + décisions Day-1 | ce commit |
| 14.C | `data.py` (replay buffer + MinAtar wrapper) + Tier 1 | |
| 14.D | `model_v1.py` (SarlQNetworkV1 + SarlSecondOrderNetworkV1) + Tier 2 | |
| 14.E | `rollout.py` (ε-greedy + cascade action selection) | |
| 14.F | `trainer.py` (DQN update + two-loss) + Tier 3 | |
| 14.G | `training_loop.py` (RL loop + checkpoint) + `evaluate.py` + Tier 4-light | |
| 14.H | `cli.py` + closeout | |
| 14.x | ACB (setting 7) — reporté | |

## Sources & specs
- Source : `external/paper_reference/sarl/maps_v1.py` (+ `PROVENANCE.md`,
  `SARL_Training_Standard.sh`). Post-mortem v1/v2 : `docs/reproduction/sarl-postmortem-20260524.md`.
- Specs : `docs/learning/reverse-prompts/domains/sarl/{data,model_v1,rollout,trainer,training_loop,losses,cli}.md`.

## Findings d'audit à reproduire sciemment (D14.2)
De `docs/reviews/audit-pre-production-20seeds-20260706.md` (section SARL) : env non seedé (T-1),
scheduler step_size=1 (T-2), cascade FO no-op settings 2/4/6 (T-3 — à vérifier sur l'archi v1,
qui cascade sur Output 1024-d), métrique = retours training ε-greedy (S-M3). Décisions Guillaume
rattachées : D1 (scheduler), D2 (bras cascade), D5 (métrique) — tranchées APRÈS reproduction.
