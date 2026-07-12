# Décisions à trancher avant le run 20 seeds × 1M (SARL / SARL+CL / MARL)

**Date :** 2026-07-06
**Pour :** Guillaume
**De :** Rémy (audit code MAPS avant relance production)
**Contexte :** audit complet du code SARL, SARL+CL, MARL avant de brûler l'allocation sur
20 seeds × 1M steps. Détail technique : `docs/reviews/audit-pre-production-20seeds-20260706.md`.
Plusieurs fixes mécaniques sont déjà faits et testés (RNG resume GPU, chemins d'agrégation).
Les points ci-dessous sont ceux qui **ne peuvent pas être tranchés sans toi** parce qu'ils
touchent la méthode ou la fidélité au papier. Rien ne part en production tant qu'ils ne sont
pas décidés.

---

## D1 — Scheduler `step_size` : 1 ou 1000 ? ⚠️ BLOQUANT (SARL + SARL+CL)

**Problème.** Le défaut de config est `StepLR(step_size=1, gamma=0.999)`, steppé à chaque
update. À `training_freq=1`, le learning rate passe sous 1 % après ~5 000 updates et atteint
numériquement 0 bien avant 1M frames. Conséquence : à 1M steps, **~99 % du run entraîne un
réseau gelé** — l'apprentissage est fini avant même la fin de l'annealing d'exploration.

**Fait établi.** Le code de référence qui a produit les chiffres du papier utilise
`step_size=1000` (`external/paper_reference/marl_tmlr/.../sarl_cl_maps.py:1235`). Le `1` est
déjà signalé dans nos notes comme typo suspectée (`D-sarl-sched-step`).

**Options.**
- (a) **`step_size=1000`** — aligne sur le code de référence. *Recommandé.*
- (b) Pilote court (1 seed, 100k frames, 1 vs 1000) pour confirmer avant de committer.
- (c) Garder `1` seulement si les auteurs confirment que c'était voulu.

**Impact si non tranché :** tout le run (SARL + SARL+CL) est potentiellement inexploitable.

---

## D2 — Bras « cascade first-order » (settings 2/4/6) : les faire tourner ?

**Problème.** Le forward first-order est déterministe, donc les 50 itérations de cascade
retournent exactement `f(x)` (valeur ET gradient). Les settings 2/4/6 sont donc
**statistiquement identiques** aux settings 1/3/5, mais coûtent **~50× plus cher** (50 forwards
par sélection d'action, par update, par cible TD). C'est documenté (`D-sarl-cascade-noop`,
tracé par run dans `metrics.json`). Seul le chemin second-order (avec dropout) fait une vraie
cascade.

**Options.**
- (a) **Ne pas courir 2/4/6 à 1M** (copier les résultats de 1/3/5) — coût ~0.
- (b) **Court-circuiter** les itérations redondantes (`effective=1`) → même résultat, même coût
  que 1/3/5, garde les 6 cellules pour l'affichage.
- (c) Rendre le first-order stochastique pour que la cascade ait un effet réel → **diverge du
  papier**.
- (d) Les courir tels quels → ~50× de node-heures pour un contraste nul a priori.

**Question de fond :** veut-on mesurer une contribution de la cascade-FO (aujourd'hui nulle par
construction) ? Si non, (a) ou (b).

---

## D3 — Sémantique des réseaux « meta » en MARL ⚠️ BLOQUANT (bras meta MARL)

**Problème.** Au rollout comme à l'update PPO, le code utilise toujours l'actor/critic
*baseline*. Les réseaux `actor_meta`/`critic_meta` ne servent **jamais à agir ni à estimer la
valeur** — ils sont entraînés uniquement par la BCE de wager, sans paramètre partagé avec le
réseau qui agit. Résultat : les **4 bras `meta=True` sont comportementalement identiques au
baseline**. Cela contredit notre propre doc d'architecture (`docs/reviews/marl-architecture.md`)
qui dit que `meta=True` doit router par `actor_meta`/`critic_meta`.

**Options.**
- (a) **Router rollout + PPO par `actor_meta`/`critic_meta`** quand `meta=True` (conforme au
  design étudiant documenté). C'est la version où « la métacognition peut changer le
  comportement/les rewards ».
- (b) Garder le wager comme **sonde passive** (lecture Pasquali & Cleeremans) et documenter que
  l'effet du meta sur les rewards (Table 7) **n'est pas testable** par ce port.

**Impact si non tranché :** le facteur « second-order » du factoriel MARL mesure du pur bruit
de seed sur l'axe reward. (Sous-point technique lié : `critic_meta` n'est actuellement jamais
entraîné — sa loss passe par `actor_meta` et son optimizer steppe à vide. À corriger dans le
sens de l'option retenue.)

---

## D4 — Que trace exactement la Figure 7 ? (SARL+CL)

**Problème.** Aucune évaluation du *catastrophic forgetting* n'existe dans le pipeline : la
validation ne tourne que sur le jeu courant du curriculum. Rien n'évalue le réseau du stage N
sur les jeux 1…N-1.

**Ce dont on a besoin de ta part :** la quantité exacte tracée par la Figure 7.
- Si ce sont des **trajectoires de forgetting pendant l'entraînement** → il faut **ajouter une
  éval multi-jeux à la boucle AVANT de lancer** (sinon la donnée n'existera jamais).
- Si c'est la **perf finale par stage** → récupérable *post-hoc* depuis les checkpoints de fin
  de stage, rien à ajouter avant le lancement.

**Impact si non tranché :** risque de finir 90 cellules × plusieurs heures et de découvrir que
la donnée pour la figure n'a pas été produite.

---

## D5 — Métrique SARL rapportée

**Problème.** La métrique actuelle = moyenne des retours sur les 100 derniers épisodes
**d'entraînement** (donc en ε-greedy, ε=0.1), dernier épisode tronqué inclus. Des résumés de
**validation greedy** sont aussi collectés mais non utilisés.

**Ce dont on a besoin :** confirmer laquelle correspond à la Table du papier (parité) — ε-greedy
training (actuel) ou validation greedy. Le biais est partagé entre settings donc le contraste
survit, mais les valeurs absolues diffèrent.

---

## D6 — Ordre des mini-batchs RNN en MARL (chunk-major vs time-major) 🔬 NOUVEAU

**Problème.** L'audit avait signalé comme bug CRITIQUE que le générateur récurrent MARL
mélange temporellement les séquences (chunk-major) alors que le RNN dé-aplatit en time-major.
**En vérifiant, j'ai trouvé que le port est FIDÈLE au papier** : la vraie fonction
`SeparatedReplayBuffer.recurrent_generator` du code de référence fait exactement le même
chunk-major. L'audit avait confondu avec deux autres fonctions non utilisées
(`naive_recurrent_generator` et la variante `shared_buffer`) qui, elles, sont en time-major.

**La vraie question (recherche).** Le papier combine un buffer chunk-major avec un RNN
time-major — c'est **peut-être un scramble latent dans le code publié lui-même**. Donc :
- (a) **Reproduire le papier tel quel** (y compris ce comportement) — fidélité maximale.
- (b) **Corriger** vers un MAPPO cohérent (time-major partout) — plus correct, mais **diverge
  des chiffres du papier**.

**Impact si non tranché :** c'est un arbitrage « reproduire vs corriger ». Par défaut je pars
sur (a) — reproduire — sauf indication contraire. (Même prudence à venir sur l'alignement des
cibles de wager MARL : je vérifie la provenance avant d'y toucher.)

---

## D7 — Target network SARL+CL : figé ou bootstrap online ? (déviation à ratifier)

**Problème.** Le port utilise un vrai target network figé (resync toutes les 500 updates). Le
code de référence, lui, bootstrappe la cible TD depuis le réseau *online* (tous les call sites
passent `policy_net` comme target ; le target global de la référence est du code mort). Le
choix du port est sans doute le **DQN correct**, mais il **diverge du papier** et change les
propriétés de stabilité.

**Options.**
- (a) Reproduire la référence (bootstrap online) — parité stricte.
- (b) Garder le target figé (plus correct) et **logger la déviation** dans `deviations.md`.

**À trancher + logger quelle que soit l'option.**

---

## Points connexes (pas des décisions, mais à savoir)

- **Seeding des environnements** (MinAtar + substrat MeltingPot) : actuellement jamais seedés
  → « seed 42 » ne reproduit pas un run. Je le corrige (fix mécanique) car le seeding est une
  règle projet ; ce sera loggé comme *amélioration de reproductibilité* vs la référence (qui ne
  seedait rien). Ça change les trajectoires par rapport aux runs non-seedés — juste pour info.
- **Nombre de seeds** : le plan dit 20, les scripts existants ont 3 à 10. Confirme le 20 pour
  qu'on dimensionne les scripts Narval en conséquence.
- **Disque plein** : le projet `ctb-gdumas85` est à 40/40 To (awolff ~19 To, gdumas85 ~16 To).
  Même après nos fixes, le run 20×1M ne pourra pas écrire ses checkpoints tant qu'il n'y a pas
  de ménage/archivage côté awolff & gdumas85 (les `/nearline` du groupe sont quasi vides).

---

## Récapitulatif express

| # | Décision | Domaine | Bloquant ? | Reco par défaut |
|---|----------|---------|-----------|-----------------|
| D1 | scheduler step_size 1→1000 | SARL, SARL+CL | **Oui** | 1000 (ou pilote) |
| D2 | bras cascade-FO 2/4/6 | SARL, SARL+CL | budget | skip / court-circuit |
| D3 | meta MARL agit ou sonde passive | MARL | **Oui** | à trancher |
| D4 | quantité exacte Figure 7 | SARL+CL | **Oui** | à préciser |
| D5 | métrique SARL (train vs valid) | SARL | non | confirmer vs papier |
| D6 | RNN chunk vs time-major | MARL | non | reproduire (a) |
| D7 | target net figé vs online | SARL+CL | non | logger déviation |
