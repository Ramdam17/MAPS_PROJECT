# MARL — décisions avant le run 20 graines × 1M (2026-07-12)

**Pour** : Guillaume, Natalie
**De** : Rémy (+ assistant)
**Contexte** : on corrige le code MARL (choix B : « faire agir la métacognition ») avant de lancer
4 substrats × 6 réglages × 20 graines × 1M pas. Ce mémo explique **ce qu'on a trouvé** et **ce qu'on
propose de corriger**, en langage simple. Il reste 2 petites questions à trancher.

---

## 1. Ce qu'on a trouvé (le vrai problème)

Dans les modèles MAPS qui **fonctionnent** (Blindsight, AGL, publiés), le montage est :

- **un seul réseau qui joue** (« le joueur ») ;
- **un juge de confiance** branché dessus, qui « parie » sur le fait que le joueur a bien répondu.

Point crucial : quand on entraîne le juge à bien évaluer sa confiance, cet entraînement **modifie aussi
le joueur** — les deux partagent le même calcul, et les corrections du juge « redescendent » dans le
joueur. C'est **par ce lien** que « se connaître » finit par changer/améliorer le comportement.
(Vérifié dans le code de référence : `blindsight_tmlr.py` et `agl_tmlr.py`, le gradient du juge remonte
bien dans le réseau qui joue avant sa mise à jour.)

**En MARL, ce lien a été coupé.** Quelqu'un a créé **deux joueurs séparés** :

- un **vrai joueur** (celui qui choisit les actions dans le jeu) ;
- un **joueur-fantôme** (`actor_meta`), sur lequel on a branché le juge.

Le juge n'influence donc **que le fantôme**, jamais le vrai joueur. Il y a même un **critique-fantôme**
(`critic_meta`) qui ne reçoit jamais aucun entraînement (poids mort).

**Conséquence directe** : activer la métacognition (réglages 3 à 6) ne change **rien** au comportement
ni aux scores par rapport à « métacognition OFF » (réglages 1-2). Le facteur « métacognition » sort
donc **nul par construction** — pas parce que MAPS ne marche pas, mais parce que le code ne le branche pas.

---

## 2. Ce qu'on propose de corriger

**Rebrancher le juge de confiance sur le VRAI joueur**, exactement comme Blindsight/AGL :

- le juge (2ᵉ ordre + pari) est nourri par les représentations **du réseau qui joue** (et non d'un
  réseau séparé) ;
- l'entraînement du juge **redescend dans le réseau qui joue** (co-entraînement), donc « méta ON »
  façonne réellement le comportement ;
- on **supprime** le joueur-fantôme et le critique-fantôme (inutiles et trompeurs).

C'est la correction la plus fidèle à l'esprit du papier : on rend MARL **cohérent** avec les domaines
où MAPS fonctionne.

**Effet attendu sur les résultats** : les réglages 3-6 (méta ON) ne seront plus identiques aux réglages
1-2. On pourra enfin comparer « avec / sans métacognition » de façon qui a du sens.

---

## 3. Deux questions annexes — TRANCHÉES (esprit du projet, faute de retour)

### (a) Ordre des séquences pour le réseau à mémoire (« RNN ») → **CORRIGÉ en « ordre du temps »**
Le code actuel range les bouts d'épisode **« par morceaux »**. **Vérifié empiriquement** : ça mélange
les séquences. Exemple : deux séquences `[10,11,12]` et `[20,21,22]` ressortent en `[10,12,21]` et
`[11,20,22]` — les pas de temps sont éparpillés entre séquences. Un réseau à mémoire nourri de séquences
mélangées **ne peut pas apprendre la dépendance temporelle** → ça vide de son sens l'usage d'un RNN.
C'est un **vrai bug** du code d'origine, pas un simple choix.
→ **Décision** : on corrige (« ordre du temps »), avec un test qui le verrouille. Indépendant de la métacognition.

### (b) Sur quoi le juge « parie » → **« l'action était-elle meilleure que prévu » (avantage > 0)**
Pas « récompense > 0 » : dans MeltingPot les récompenses sont **rares**, la cible serait quasi toujours
« non » → le juge apprendrait à toujours parier bas → signal inutile. On prend l'équivalent RL propre de
« ai-je bien décidé ? » = **l'avantage** (l'action a-t-elle rapporté plus que la moyenne attendue). Il est
**déjà calculé** par l'algo, **équilibré** (~moitié positif) et **dense** (à chaque pas) → vrai signal.
→ **Décision** : cible = one-hot sur (avantage > 0), sur les 2 unités de pari (constante MAPS).

---

## 4. Ce qui ne change pas

- Matrice : **4 substrats × 6 réglages × 20 graines × 1M pas** = 480 runs.
- Réglages 1-6 = le factoriel habituel (cascade ON/OFF × méta ON/OFF).
- Format de sortie = un fichier JSON par graine, **même structure** que les fichiers existants
  (retours par épisode, longueurs, validations, ratio de pari, etc.).

---

## Décisions prises (résumé)

> Guillaume/Natalie ne tranchant pas au cas par cas, ces choix sont pris côté équipe technique, dans
> l'esprit du projet (implémenter une métacognition qui agit, pas reproduire la lettre du papier).
> Réversibles si vous n'êtes pas d'accord.

1. **Correction principale** : rebrancher le juge sur le vrai joueur (co-entraînement, façon Blindsight/AGL),
   supprimer l'acteur-fantôme et le critique-fantôme. **Monitoring uniquement** (pas de contrôle direct de
   l'action) — cohérent avec les autres domaines. ✅ retenu.
2. **Ordre des séquences RNN** : corrigé en « ordre du temps » (le « par morceaux » actuel mélange les
   séquences — vérifié). ✅ retenu.
3. **Cible du pari** : (avantage > 0), pas (récompense > 0) — dense et équilibré vs quasi-toujours-nul. ✅ retenu.
4. **(mineur) Agrégation 16 agents** pour le rapport : `episode_returns` = moyenne par agent (même échelle
   que les fichiers SARL) + total collectif à part.
