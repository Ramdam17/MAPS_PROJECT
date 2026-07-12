# Sprint 16 — MARL rewrite (env-independent core tested; env/runner code-only)

**Status:** 🟢 core done + env/runner/cli vendored (2026-07-12) — voir Closeout en bas.
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

---

## Closeout (2026-07-12)

### Ce qui a été construit (`refactor/marl`, rien poussé)
Cœur MARL indépendant de l'env, **testé en parité/structure**, + couche env/runner/cli
**vendored code-only**. 10 commits (16.A → 16.H). Suite `tests/unit/domains/marl/` :
**45 passés / 2 skippés** (1 skip = `ppo_update` GPU-only, 1 skip = import env sans meltingpot).

| Fichier | Rôle | Commit | Tests |
|---------|------|--------|-------|
| `data.py` | SeparatedReplayBuffer (insert, GAE, recurrent_generator **chunk-major** M-C1) | b51e618 | 5 |
| `valuenorm.py` | ValueNorm (EMA débiaisée) | 886cf23 | 4 |
| `util.py` + `encoder.py` + `rnn.py` | init/check/conv-params ; CNNBase (valid conv) ; RNNLayer (GRU+LN, cascade séparée M-H3) | cd5c669 | 4 |
| `act.py` | ACTLayer Discrete (asymétrie `.log_prob`/`.log_probs` fidèle) | 0bb9fb5 | 5 |
| `policy.py` | R_Actor + R_Critic (chemin GRU) | ba7c039 | 4 |
| `policy_meta.py` | SecondOrderNetwork + RNNLayer_Meta + R_Actor_Meta + R_Critic_Meta (poids mort M-C3) | 499a0bd | 7 |
| `rmappo_policy.py` + `trainer.py` | wrapper 4 réseaux/4 optimizers + R_MAPPO (cal_value_loss, ppo_update, train) | 43897d8 | 9 (+1 gpu skip) |
| `env.py` + `base_runner.py` + `runner.py` + `cli.py` | **vendored** MeltingPot + boucle d'entraînement + entrée | 1e8df81 | 7 (+1 skip) |

### Détails fidèles verrouillés (à ne pas « corriger » sans décision)
- **rnn_cells (LSTM) retirés de bout en bout** (chemin GRU, D16) : buffer → réseaux → wrapper →
  trainer manipulent le **tuple à 12 éléments** (sans cells). La source en a 14.
- **`ppo_update` épingle `.cuda()`** sur les tenseurs du wager (`r_mappo.py:162-163,220`) *après*
  `.to(**tpdv)` → no-op sur GPU, plante sur CPU. **Reproduit verbatim** → `ppo_update`/`train`
  = GPU only ; `cal_value_loss` + construction + evaluate_actions/_meta testés sur CPU.
- **M-C3 poids mort** : `evaluate_actions_meta` lit `actor_meta` des deux côtés → `critic_meta`
  jamais entraîné, son `.step()` est un no-op (reproduit).
- **Correction de note d'audit** : le vrai `get_wager_objective` (base_runner) n'est **pas**
  un simple `reward>0` : c'est un encodage 2-classes `[reward//100, 1]` si `reward > comparaison`
  sinon `[1, reward//100]`, complété par `[0,0]` jusqu'à `episode_length`. (La cible BCE est
  donc de forme `(episode_length, 2)`.)
- **`check` du trainer** = celui de `algorithms/utils/util.py` (retourne l'entrée si pas ndarray),
  PAS celui de `utils/util.py` (qui renvoie `None`) → notre `marl/util.check` est le bon.

### Couche vendored (choix utilisateur : « copie quasi à l'identique »)
`env.py`/`base_runner.py`/`runner.py`/`cli.py` recopiés verbatim depuis `paper_reference`,
`# ruff: noqa` en tête (pas de reformat). Seuls changements, documentés dans chaque en-tête :
- chemins d'import `onpolicy.*` → `maps.domains.marl.*` ;
- `base_runner.__init__` : 3 sites de construction (Policy/TrainAlgo/buffer) recâblés vers nos
  constructeurs à mots-clés (seule exception inévitable à « imports seulement », car nos modules
  ont abandonné `args` argparse) ;
- **`energy_tracker` ABSENT de la référence** (module du repo étudiant, pas du code de référence) →
  **stubs inertes** (no-op) pour garder les sites d'instrumentation verbatim ; le suivi d'énergie
  ne fait donc rien ;
- `cli.py` garde le factoriel MAPS (setting 1-6 → meta × cascade1/2) ; la plomberie générique
  (`config.get_config`, `env_wrappers`, runner *shared*) reste sur `onpolicy.*` (non portée —
  hors périmètre du rebuild par domaine).

`runner.py`/`base_runner.py` s'importent localement (aucune dépendance meltingpot à l'import) →
testés en structure. `env.py` a besoin de dmlab2d (test skip). `env.py`/`cli.py` byte-compilent.

### Reporté / à trancher plus tard
- **Décisions Guillaume D3 (meta sonde vs agissant) / D6 (chunk vs time-major)** : la source =
  sonde + chunk-major (reproduits). Version *corrigée* éventuelle = après validation repro.
- **Parité numérique bout-en-bout** : impossible ici (meltingpot/dmlab2d + GPU absents). À faire
  sur le cluster : lancer `cli.py` sur 1 substrat, comparer aux logs de référence.
- **Plomberie générique** (`config.py`, `env_wrappers.py`, runner *shared*) non portée — à
  vendoriser si on veut un `cli.py` autonome hors `external/`.
- **conftest `torch.set_num_threads(1)`** (D16.6) : déjà présent (hérité du conftest partagé).
