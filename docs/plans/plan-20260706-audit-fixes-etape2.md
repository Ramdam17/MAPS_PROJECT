# Plan: Fixes mécaniques post-audit (Étape 2)

**Date:** 2026-07-06
**Référence:** `docs/reviews/audit-pre-production-20seeds-20260706.md` (§4 étape 2)
**Estimated complexity:** L

---

## Problem Statement

L'audit pré-production a identifié un ensemble de défauts mécaniques, sans ambiguïté
scientifique, qui doivent être corrigés avant le run 20 seeds × 1M steps : crash du resume
GPU SARL, environnements jamais seedés, bug d'ordre temporel dans le générateur récurrent
MARL, cibles wager désalignées, chemins d'agrégation incohérents, `--requeue` sans
`--resume`, et guards manquants (NaN, resume, provenance). Tous les sites ont été
re-vérifiés à la ligne près avant ce plan.

**Hors scope (décisions Guillaume, étape 1) :** scheduler step_size, bras cascade-FO no-op,
routing des réseaux meta MARL (M-C2) et donc la résolution finale de `critic_meta` (M-C3),
métrique SARL, exigences Figure 7.

## Scientific Rationale

Aucun de ces fixes ne change la méthode : ils rendent le code conforme à son intention
documentée (commentaires/docstrings contredits par le code) ou aux règles du projet
(seed control obligatoire, CI sur les agrégats). Les changements qui altèrent les
trajectoires (seeding env) sont des améliorations de reproductibilité loggées dans
`deviations.md` — le code de référence étudiant ne seedait rien.

## Implementation Approach

Ordre d'implémentation (dépendances explicites) :

### F1 — S-C1 : fix RNG restore GPU dans SARL
- `src/maps/experiments/sarl/training_loop.py:505-507` : porter le fix SARL+CL verbatim
  (`.cpu().byte()` sur `rng_torch`, liste-comprehension sur `rng_torch_cuda`), avec le même
  commentaire explicatif que `sarl_cl/training_loop.py:548-552`.
- Test : miroir du test SARL+CL s'il existe ; sinon, test CPU qui simule la condition
  (caster `payload["rng_torch"]` en float32 après save, vérifier que le restore réussit).

### F2 — T-1a : seeding MinAtar + état env dans les checkpoints (SARL, SARL+CL)
- `scripts/run_sarl.py` et `scripts/run_sarl_cl.py` : `env.seed(effective_seed)` juste après
  `_build_env` (l'API `Environment.seed()` existe : `external/MinAtar/minatar/environment.py:39-42`
  — vérifier qu'elle propage bien au RandomState du jeu, sinon seeder les deux niveaux).
- Checkpoints : ajouter `rng_env_wrapper` (`env.random.get_state()`) et `rng_env_game`
  (`env.env.random.get_state()`) aux payloads ; restore symétrique. L'env est construit dans
  le script, pas dans la boucle → passer l'env (ou ses états RNG) à
  `_persist_checkpoint`/`_restore_from_checkpoint` — inspecter la signature réelle avant de
  choisir le mécanisme le moins invasif.
- Bump `_CHECKPOINT_FORMAT_VERSION` et `_CHECKPOINT_FORMAT_VERSION_CL`.
- Logger `sticky_action_prob`/`difficulty_ramping` effectifs au démarrage.
- Entrée `deviations.md` : la référence étudiante ne seedait pas l'env.

### F3 — M-C1 : ordre time-major dans `recurrent_generator`
- `src/maps/experiments/marl/data.py:305-314` : `np.stack(..., axis=1)` pour les 10 tableaux
  séquentiels (PAS `rnn_states_batch`/`rnn_states_critic_batch`, qui restent (B, N, H) —
  un état initial par chunk).
- Test : buffer rempli avec `arange`, vérifier qu'après le `view(episode_len, batch_num, -1)`
  de `rnn.py:61` chaque colonne est une tranche temporelle contiguë d'un seul chunk.

### F4 — M-C4 : cibles wager alignées sur les mini-batches
- `runner.py:489-501` : stocker les cibles wager par pas dans le buffer (`(T, N, A, 2)`,
  même layout que `actions`) au lieu du side-channel `(A, T*N, 2)`.
- `data.py:recurrent_generator` : `_cast` + découpage par chunk comme `actions`, yield d'un
  13e champ (ou champ optionnel) `wager_targets_batch`.
- `trainer.py:294-302, 357-366` : consommer la cible par batch au lieu du tableau global.
- Garde-fou : `assert` de shape entre prédiction et cible (le bug actuel ne survivait que
  par coïncidence de shapes).
- Test : cibles arange, vérifier l'alignement (t, n) prédiction/cible après shuffle.
- NOTE : ne PAS toucher au routing actor_meta/critic_meta (M-C2/M-C3, décision Guillaume).
  Le fix aligne seulement les cibles de la loss existante.

### F5 — S-H2 : réconcilier run_sarl ↔ aggregate_sarl
- `scripts/aggregate_sarl.py:128` : base par défaut = `paths.scratch_root/maps/outputs/sarl`
  (même expression que `run_sarl.py:205-215`) + option `--base-dir` pour surcharger.
- Provenance (S-M3) : à la collecte de chaque cellule, vérifier `seed`/`game` de
  `metrics.json` contre le répertoire, et `total_frames` homogène entre cellules + option
  `--expected-frames` ; échec bruyant sinon.

### F6 — S-H1 : `--resume` dans `sarl_array.sh` (STRICTEMENT après F1)
- `scripts/slurm/sarl_array.sh:102-107` : ajouter `--resume` à l'invocation.
- Rafraîchir le header WARNING stale (lignes 32-47) qui décrit l'état pré-checkpointing.

### F7 — Guards NaN (SARL + SARL+CL)
- `sarl/training_loop.py:675-678` et équivalent CL : `math.isfinite` sur la loss ;
  si non-fini → log ERROR avec contexte (t, episode, update) + abort (exception dédiée).
  Pas de silencing : on loggue tout et on meurt bruyamment.

### F8 — Guards de resume élargis + provenance metrics.json
- `_CHECKPOINT_CFG_GUARDS` (SARL) et `_CHECKPOINT_CFG_GUARDS_CL` : ajouter les
  hyperparamètres numériquement porteurs (batch_size, gamma, alpha, lrs, scheduler_*,
  target_update_freq, replay_buffer_size, poids de loss CL, teacher_load_path).
- `metrics.json` (SARL + SARL+CL) : dumper le `asdict(cfg)` complet (Paths → str).

### F9 — T-1b : seeding du substrat MeltingPot (investigation)
- Unknown : l'API `substrate.build` accepte-t-elle un seed proprement ?
  → lire `external/meltingpot` + `marl/env.py:441` ; si oui, thread `--seed` ; si non,
  documenter l'impossibilité dans `deviations.md` et logger le fait au démarrage.

### Config changes needed
Aucun nouveau paramètre YAML obligatoire. `--expected-frames` (CLI aggregate) et le seed env
dérivent de l'existant.

## Risks & Unknowns
- **Signature checkpoint SARL/CL (F2)** : le passage de l'état env peut toucher plusieurs
  call sites → choisir la forme la moins invasive après lecture des signatures réelles.
- **F3/F4 changent les numériques MARL** : c'est le but (l'ancien comportement est un bug),
  mais tout smoke antérieur (E.17b3) n'est plus comparable → re-smoke obligatoire (étape 4
  de l'audit) avant production.
- **MinAtar `seed()` (F2)** : vérifier si le wrapper re-seede le RandomState du jeu ou
  seulement le sien — sinon seeder les deux explicitement.
- **F9** : peut s'avérer non trivial (Lua/C++) ; borné à investigation + documentation.

## Verification Plan
- [ ] `uv run pytest` complet vert (dont parité SARL Tier-1/2/3 inchangée pour F1/F7/F8)
- [ ] Nouveaux tests unitaires : F1 (RNG dtype), F3 (ordre temporel), F4 (alignement wager),
      F5 (provenance), F8 (guard déclenche sur mismatch)
- [ ] `uv run ruff check .` + `ruff format --check`
- [ ] Round-trip save/restore checkpoint SARL avec env RNG : deux runs seedés identiques
      → trajectoires identiques (test lent, CPU, petit budget de frames)
- [ ] Sanity : `sarl_array.sh` relu ligne à ligne après édition (pas de test automatique)

## Definition of Done
- [ ] Tous les tests passent
- [ ] `deviations.md` mis à jour (env seeding, guards)
- [ ] Versions de format checkpoint bumpées là où le payload change
- [ ] Commits scopés par fix (F1..F9), référençant ce plan
- [ ] Rapport d'audit §4 étape 2 cochable ; étapes 3 (infra Narval) et 4 (re-validation)
      restent à faire ensuite

---

## Journal d'implémentation

### 2026-07-06 — F1 fait, F5 fait, F3 RECLASSÉ (décision, pas mécanique)

**Fait & commité :**
- **F1** (`dda7529`) — fix `.cpu().byte()` du restore RNG GPU dans `sarl/training_loop.py`
  + test de régression qui échoue sans le fix. 40/40 tests SARL verts.
- **F5** (`30cfa6d`) — base d'`aggregate_sarl.py` alignée sur le writer + checks de
  provenance (game/seed/frames) + `--base-dir`/`--expected-frames`. 5 tests, verts.

**F3 — RETOURNEMENT : ce n'est PAS un fix mécanique. Escaladé en décision.**

En implémentant F3 (passer les stacks séquentiels du générateur récurrent en `axis=1`),
mon test de contiguïté temporelle passait MAIS la parité `test_recurrent_generator_matches_student_seeded`
cassait. Investigation :

- Le port cible `SeparatedReplayBuffer.recurrent_generator`. Dans le vrai code du papier
  (`external/paper_reference/marl_tmlr/onpolicy/utils/separated_buffer.py:352+`) cette
  fonction fait `np.stack(x)` (**axis=0**) puis `_flatten(L, N, x)` = `x.reshape(T*N, ...)`
  **sans transpose** (l.10-11) → ordre **chunk-major**. Le port ET le `_student_ref` vendu
  reproduisent exactement ça. **Le port est fidèle au papier.**
- Le `axis=1` (time-major) que l'audit M-C1 citait comme « la référence » provient de deux
  AUTRES fonctions que le port n'utilise pas : `naive_recurrent_generator` (l.318-328) et la
  variante `shared_buffer`. **L'audit a confondu les variantes separated vs shared/naive.**
- Reste ouverte une vraie question de recherche : le papier combine un buffer chunk-major
  avec un RNN qui dé-aplatit en time-major (`rnn.py:42` `view(episode_len, batch_num)`).
  Est-ce un scramble latent DANS le code publié ? Possible — mais « corriger » = **diverger
  de la méthode publiée**. C'est une décision (comme scheduler/cascade), pas un fix mécanique.

→ **F3 déplacé de l'étape 2 (mécanique) vers l'étape 1 (décisions Guillaume).** data.py et
son test ont été **revertés** ; l'arbre est propre.

**Conséquence pour F4 (cibles wager) :** même prudence. Avant de toucher, vérifier comment
le papier (`separated_buffer` + trainer de référence) fait circuler les cibles de wager —
le port a peut-être introduit le side-channel `wager_per_agent`, ou l'a copié. Ne PAS
traiter F4 comme mécanique sans cette vérification de provenance.

**Environnement :** `.venv` projet corrompu (disque plein) supprimé + recréé en symlink vers
`/scratch/rram17/venvs/maps-project` (convention Sprint-08 A.2). Tests tournent via cet interpréteur.
Note : la suite complète `tests/unit/experiments/marl/` **hang sur le 6e test de
`test_checkpoint.py`** (pré-existant, indépendant de tout changement de ce plan) — à
investiguer séparément.
