# Sprint 16 — MARL rewrite (env-independent core tested; env/runner code-only)

**Status:** 🔵 open (2026-07-12)
**Branch:** `refactor/marl` (branchée depuis `6f07916` — cœur partagé complet)
**Owner:** Rémy Ramadour
**Depends on:** Sprint 11 ✅ (core/) + Sprints 12-15 ✅ (patron domaine)
**Numerical reference:** `external/paper_reference/marl_tmlr/onpolicy/` (MAPPO + MAPS meta ;
variante **separated** — c'est celle que `meltingpot_runner.py` utilise ; §5, Table 7).

---

## Context

MARL = MeltingPot 2.0 MAPPO + briques MAPS. Le plus complexe (RL multi-agents on-policy,
GAE, RNN, valuenorm). `meltingpot`/`dmlab2d` sont des paquets natifs **absents** du venv
scratch (install manuelle Linux) → parité limitée au code indépendant de l'env.

## Investigation Day-1 (source `marl_tmlr/onpolicy`) — les 4 « bugs » d'audit

Lecture de la source : les 4 findings CRITIQUES de l'audit MARL sont des **propriétés du
code du papier**, pas des bugs de port. Le rebuild fidèle les **reproduit et documente** ;
une éventuelle version *corrigée* relève des décisions Guillaume D3/D6 (plus tard).

- **M-C2 (meta n'agit jamais)** : `collect()` joue via `get_actions` → `self.actor` baseline
  seul (rMAPPOPolicy L293/430). Le meta est calculé à part (`policy.meta()`) comme **sonde
  de wager** entraînée par BCE (`wager_objective = reward>0`, meltingpot_runner L338).
  L'env est joué par l'acteur baseline → **les bras meta ont le même comportement/reward que
  baseline**. Design du papier ; le meta MARL est un *read-out de wager*, pas un modificateur
  de politique.
- **M-C3 (critic_meta non entraîné)** : `evaluate_actions_meta` (L405) utilise `actor_meta`
  pour les DEUX côtés (actor + critic). `critic_meta` ne reçoit jamais de gradient →
  `critic_meta_optimizer.step()` (r_mappo L233) est un **no-op dans la source**. Poids mort.
- **M-C1 (ordre RNN chunk-major)** : `separated_buffer.recurrent_generator` = `np.stack`
  (axis 0) + `_flatten` (reshape sans transpose) → **chunk-major** dans la source. Fidèle =
  chunk-major. (Corrige le quasi-faux-pas de la session F3 : ne PAS forcer time-major.)
- **M-C4 (cibles wager)** : `get_wager_objective` construit par pas, passé à `train()`.
  Reproduire ; vérifier l'alignement de shape au portage.

## Décisions Day-1 (verrouillées 2026-07-12)

### D16.1 — Source & variante
`marl_tmlr/onpolicy`, variante **separated** (SeparatedReplayBuffer, meltingpot_runner).

### D16.2 — Baseline fidèle (bugs/quirks du papier reproduits, documentés)
Reproduire M-C1/C2/C3/C4 comme propriétés du papier. Pas de correction (D3/D6 = plus tard).

### D16.3 — meltingpot absent → parité sur le cœur indépendant de l'env
Testé en parité : data/buffer (recurrent_generator, GAE), valuenorm, encoder, rnn, act,
policy (R_Actor/R_Critic + meta), trainer (MAPPO update). Env-wrapper + runner + cli : portés
en code, tests marqués `linux_only`/skip (non exécutables sans meltingpot).

### D16.4 — Périmètre (scope choisi 2026-07-12)
Cœur testé maintenant ; env/runner/cli code-only.

### D16.5 — Schéma settings & substrats
6-cell (cascade × meta), substrats depuis `config/domains/marl/env/` (commons_harvest, etc.).

### D16.6 — conftest `torch.set_num_threads(1)`
Re-appliquer sur cette branche (cf. Sprint 15 — contention threads).

---

## Plan de phases
| Phase | Contenu | Testable ici |
|-------|---------|--------------|
| 16.A | ouvrir sprint + décisions | — |
| 16.C | `data.py` (SeparatedReplayBuffer : insert, GAE, recurrent_generator M-C1) | ✅ Tier 1 |
| 16.D | `valuenorm.py` + `util.py` | ✅ |
| 16.E | `encoder.py` + `rnn.py` (RNN chunk handling) | ✅ Tier 2 |
| 16.F | `act.py` + `policy.py` (R_Actor/R_Critic + meta) | ✅ Tier 2 |
| 16.G | `trainer.py` (MAPPO update : GAE, clip, valuenorm, meta wager M-C2/C3/C4) | ✅ Tier 3 |
| 16.H | `env.py` + `runner.py` + `cli.py` (code-only, tests linux_only) + closeout | ⚠️ skip |

## Findings d'audit / décisions Guillaume rattachés
M-C1/C2/C3/C4 (reproduits, documentés), M-H1 (substrat non seedé), M-H2 (cascade rollout vs
train), M-H4 (troncature=terminaison). Guillaume : D3 (meta sonde vs agissant — la source =
sonde), D6 (RNN chunk vs time-major — la source = chunk-major).
