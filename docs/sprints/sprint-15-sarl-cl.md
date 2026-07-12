# Sprint 15 — SARL+CL rewrite + tests (CL mechanics first)

**Status:** 🔵 open (2026-07-12)
**Branch:** `refactor/sarl_cl` (branchée depuis `6f07916` — cœur partagé complet)
**Owner:** Rémy Ramadour
**Depends on:** Sprint 11 ✅ (core/) + Sprint 14 ✅ (SARL v1 — modèle/loss réutilisés en esprit)
**Numerical reference:** `external/paper_reference/sarl_cl_maps.py` (sur-ensemble de
`maps_v1.py` avec la machinerie continual-learning ; §4, Figure 7).

---

## Context

SARL+CL = SARL (MinAtar DQN v1) + apprentissage continu multi-jeux : un réseau
apprend une séquence de jeux, avec un **teacher gelé** (le réseau du jeu précédent)
qui ancre l'apprentissage pour limiter l'oubli catastrophique. Ajoute 3 termes de
perte au DQN : `task` (courant), `distillation` (ancre L2 vs teacher), `feature`
(MSE des activations intermédiaires vs teacher).

Auto-contenu depuis `6f07916` : les réseaux sont **dupliqués intentionnellement**
(spec model.md — structurellement proches de SARL v1 mais classes distinctes) +
le `AdaptiveQNetwork` (adaptateur 1×1 + zero-pad channels) pour le transfert
cross-jeux. Ne dépend pas de `domains/sarl`.

## Goal

La **machinerie CL** paper-faithful vs `sarl_cl_maps.py` (bugs compris) : modèle
(dont AdaptiveQNetwork), pondération dynamique des pertes, pas d'update CL
(task+distill+feature, two-loss, teacher gelé), boucle d'entraînement. L'évaluation
complète de l'oubli (Figure 7) est **différée** (dépend de la décision D4 de Guillaume).

---

## Décisions Day-1 (verrouillées 2026-07-12)

### D15.1 — Source & variante
`external/paper_reference/sarl_cl_maps.py`, réseaux **v1-style** (SarlCLQNetwork = archi
v1 ; comparison_layer 1024×1024 actif ; cascade sur Output).

### D15.2 — Baseline fidèle (bugs compris, comme D13.2/D14.2)
Reproduire sarl_cl_maps.py tel quel (env non seedé, scheduler, cascade). **Pas** F1/F5.

### D15.3 — Branche auto-contenue + réseaux dupliqués (spec)
`domains/sarl_cl/` a ses propres `data.py` + `model.py` (SarlCLQNetwork,
SarlCLSecondOrderNetwork, AdaptiveQNetwork). Duplication assumée vs `domains/sarl`
(reproductions indépendantes par domaine).

### D15.4 — « Distillation » = ancre L2 `weight_regularization` (cœur), PAS Hinton KL
Le terme `distillation` actif de la source est `compute_weight_regularization` (L2,
EWC-style). Le `DistillationLoss` (KL) est instancié mais **jamais appelé** — le cœur
l'a déjà droppé (D11.5) et fournit `weight_regularization`. On l'utilise.

### D15.5 — Poids de mélange = **0.4 / 0.4 / 0.2** (task / distill / feature)
= défaut argparse de la source (`weight1=40, weight2=40, weight3=20`, /100). C'est la
valeur **fidèle** ; tranche le finding d'audit CL-M1 (l'ancien défaut de dataclass
0.3/0.6/0.1 divergeait de la source). Overridable.

### D15.6 — Périmètre : mécanique CL d'abord, curriculum complet différé
Porter : data, model (+AdaptiveQNetwork), loss_weighting (DynamicLossWeighter +
weight_reg + feature), pas d'update CL, boucle d'entraînement (1-stage / petit
curriculum), cli. **Différé** : orchestration multi-jeux complète, `curriculum_evaluation`,
éval cross-jeux de l'oubli (Figure 7) — dépend de la décision D4 de Guillaume (ce que
trace exactement la Fig 7). Findings audit CL-H2 (handoff)/CL-H3 (éval forgetting) → avec.

### D15.7 — Cœur : re-ajouter `cae_loss(recon="huber")`
Le FO task loss CL est CAE-Huber (comme SARL). L'extension huber du cœur ne vit que sur
`refactor/sarl` → à re-appliquer ici (branches indépendantes — cf. plan « bug de cœur
corrigé N× »).

### Parité (adaptée RL+CL)
Tier 1 (data — buffer/state/wager, réutilisé). Tier 2 (AdaptiveQNetwork forward +
channel-adapt/zero-pad + les pièces v1). Tier 3 (pas d'update CL : task+distill+feature,
two-loss, vs réplique inline source). Tier 4-light (court run avec teacher gelé : ancres
weight-reg actives, déterminisme d'orchestration).

---

## Plan de phases
| Phase | Contenu | Commit |
|-------|---------|--------|
| 15.A | ouvrir sprint + décisions Day-1 | ce commit |
| 15.C | `data.py` + cœur `cae_loss(huber)` re-ajouté + Tier 1 | |
| 15.D | `model.py` (SarlCLQNetwork, SarlCLSecondOrderNetwork, AdaptiveQNetwork) + Tier 2 | |
| 15.E | `loss_weighting.py` (DynamicLossWeighter + LossMixingWeights) | |
| 15.F | `trainer.py` (sarl_cl_update_step : task+distill+feature) + Tier 3 | |
| 15.G | `training_loop.py` + `cli.py` + Tier 4-light | |
| 15.H | closeout | |

## Findings d'audit à reproduire / rattacher
CL-C1 (scheduler), CL-C2 (env non seedé), CL-H1 (target-net binding — à vérifier sur
sarl_cl_maps.py), CL-H2 (handoff curriculum), CL-H3 (pas d'éval forgetting → différé D15.6).
Décisions Guillaume : D4 (Figure 7), D7 (target net) — après reproduction.
