# Plan — reconstruction par domaine (1 branche = cœur partagé + 1 domaine)

**Date :** 2026-07-12
**Principe :** reproduction FIDÈLE depuis `external/paper_reference/`, bugs compris.
Aucune correction. Les bugs connus (audit) sont *documentés et reproduits sciemment* ;
les décisions (mémo Guillaume) sont rattachées aux domaines et tranchées PLUS TARD,
quand on navigue le code ensemble.

## Topologie des branches

Base commune = **`6f07916`** (fin Sprint 12.C) : cœur partagé **complet** —
`core/` (cascade, second_order, losses), `utils/` (config, device, logging, paths,
seeding), `networks/first_order_mlp.py` — tous les domaines en stubs, config + specs
déjà posées, **aucun code Blindsight** (les fichiers `domains/blindsight/{data,trainer,
cli,augmentations}` arrivent en 12.D-H, après ce commit).

> Correction de plan (2026-07-12) : la base n'est PAS `c358fe7` (fin Sprint 11.6).
> À ce commit il manque le complément `utils/` (12.B) et `first_order_mlp.py` (12.C),
> pourtant partagés (AGL les réutilise). Vérifié : `diff 6f07916 ↔ refactor/blindsight`
> sur `core/utils/networks` est vide → `6f07916` porte le cœur partagé final.

```
6f07916  (cœur partagé complet : core + utils + networks + stubs)
 ├─ refactor/blindsight   Sprint 12   ✅ FAIT (le patron, = base + domains/blindsight)
 ├─ refactor/agl          Sprint 13   → créer
 ├─ refactor/sarl         Sprint 14   → créer
 ├─ refactor/sarl_cl      Sprint 15   → créer
 └─ refactor/marl         Sprint 16   → créer   (METTA reporté)
```

Chaque branche part de `6f07916` et ne remplit QUE `src/maps/domains/<son_domaine>/`
(+ ses tests + rien d'autre). Cœur identique bit-for-bit sur les 5 branches.
**Conséquence maintenance :** un futur bug de cœur = à corriger 5×. Accepté pour la
phase repro ; réunification éventuelle plus tard.

## Le patron (repris de Sprint 12 Blindsight), par phases

Pour chaque domaine X :
- **A — ouvrir le sprint** : décisions Day-1 lockées ; source = `paper_reference/<X>`.
- **B — cœur/utils/networks** : déjà là, réutilisés tels quels (rien à faire).
- **C…F — remplir `domains/X/`** : porter verbatim depuis `paper_reference`, guidé par les
  specs `docs/learning/reverse-prompts/domains/X/*.md` (déjà écrites).
- **G — parité 4 niveaux** vs `paper_reference`, bit-exact :
  1. donnée, 2. réseau (forward + 50-cascade + backward), 3. un pas d'update
  (loss + backward + step + scheduler), 4. boucle courte (n_epochs=2) end-to-end.
- **H — walkthrough + closeout** : + section « bugs connus reproduits » (audit) +
  « décisions rattachées » (mémo).

## Détail par domaine

### AGL — Sprint 13 (le plus simple, rode le patron)
- **Source :** `paper_reference/agl_tmlr.py`. **Specs :** 4 (data, pool, trainer, cli).
- **Fichiers :** `domains/agl/{data, pool, trainer, cli}.py`.
- **Risque clé (déjà noté Sprint 12) :** `D-agl-reset` — reset du first-order après
  pré-entraînement (mécanisme de la dissociation conscient/inconscient AGL). À reproduire
  fidèlement.
- **Dette audit :** aucune (AGL hors périmètre de l'audit SARL/MARL). Bon domaine pour
  valider le patron de bout en bout.

### SARL — Sprint 14
- **Source :** `paper_reference/sarl_maps.py` (+ `sarl_ac_lambda.py` pour setting 7 ACB).
  **Specs :** 8. **Fichiers :** `domains/sarl/{data, model, rollout, trainer, evaluate,
  cli}.py` + dépendance MinAtar.
- **Bugs connus à reproduire SCIEMMENT (audit) :**
  - env MinAtar jamais seedé (T-1)
  - scheduler `step_size=1` qui gèle le LR (T-2)
  - cascade first-order = no-op → settings 2/4/6 ≡ 1/3/5 (T-3)
  - métrique = retours d'entraînement ε-greedy (S-M3)
  - **NE PAS** porter les fixes F1 (RNG resume GPU) ni F5 (chemins aggregate) — repro fidèle.
- **Décisions rattachées :** D1 (scheduler), D2 (bras cascade), D5 (métrique).

### SARL+CL — Sprint 15
- **Source :** `paper_reference/sarl_cl_maps.py`. **Specs :** 5.
  **Fichiers :** `domains/sarl_cl/{model, trainer, loss_weighting, training_loop, cli}.py`
  + teacher/curriculum.
- **Bugs connus à reproduire :** scheduler (CL-C1), env non seedé (CL-C2), target-net
  bootstrap online vs figé (CL-H1), handoff curriculum incomplet (CL-H2), aucune éval de
  forgetting (CL-H3).
- **Décisions rattachées :** D4 (Figure 7 / forgetting), D7 (target net).

### MARL — Sprint 16 (le plus complexe)
- **Source :** `paper_reference/marl_tmlr/onpolicy/` (SeparatedReplayBuffer, r_mappo,
  r_actor_critic + r_actor_critic_meta). **Specs :** 12.
  **Fichiers :** `domains/marl/{env, data, encoder, rnn, policy, act, trainer, runner,
  valuenorm, cli}.py` + meltingpot.
- **Bugs connus à reproduire :** ordre RNN chunk-major / mélange temporel (D6), meta jamais
  branché au comportement (M-C2), critic_meta non entraîné (M-C3), cibles wager désalignées
  (M-C4), substrat non seedé, cascade rollout ≠ training (M-H2), troncature=terminaison (M-H4).
- **Décisions rattachées :** D3 (meta branché ou sonde), D6 (reproduire vs corriger le mélange).
- **Note parité :** plus lourde (env meltingpot). Cibler au minimum parité buffer + update +
  rnn (comme le `_student_ref` actuel), le train-loop complet en 4-light si faisable.

## Definition of Done (par domaine)
- Parité 4 niveaux bit-exact vs `paper_reference`.
- La branche ne contient QUE `core/` + un seul `domains/X/` rempli (les autres restent stubs).
- Closeout : bugs connus reproduits + décisions rattachées + insights.
- Tests unit + parity verts dans le venv scratch.

## Logistique
- **Env :** venv scratch en place (`/scratch/rram17/venvs/maps-project`), SARL+MARL testables.
- **Disque projet plein :** bloque les *runs* empiriques (pas la repro ni les tests). À régler
  avant tout lancement de production.
- **Préserver l'audit :** `docs/reviews/audit-pre-production-20seeds-20260706.md` +
  `decisions-guillaume-20260706.md` vivent sur la branche locale condamnée `wip/centralize`.
  À recopier sur la nouvelle base, puis à replier par domaine dans les closeouts (réponse Q3).

## Hors périmètre / à ne pas faire maintenant
- Pas de création de branches ni de code domaine avant validation de ce plan ensemble.
- Aucune correction de bug (repro fidèle). F1/F5 abandonnés.
- METTA reporté.

## Ordre d'exécution retenu
AGL (13) → SARL (14) → SARL+CL (15) → MARL (16).
