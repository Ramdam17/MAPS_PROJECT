# Fix (a) — rebrancher le juge de confiance sur le vrai acteur (co-entraînement)

**Domaine :** MARL (MeltingPot MAPPO + MAPS)
**Branche :** `run/marl-20seeds-1M` — moteur **neuf** `src/maps/domains/marl/`
**Statut :** PLAN (à implémenter après confirmation baseline GPU)
**Contexte :** run de prod 20 graines × 1M ; « implémenter l'esprit du projet », pas la lettre du papier.

---

## 1. Le problème (confirmé au niveau ligne, dans les DEUX moteurs)

Dans MAPS Blindsight/AGL (qui fonctionnent, publiés) : **un seul réseau joue**, un **juge de
confiance** (2ᵉ ordre + pari) est branché sur SES représentations, et l'entraînement du juge
**redescend dans le réseau qui joue** (gradient partagé). C'est ce lien qui fait que « se connaître »
façonne le comportement.

En MARL, le lien est coupé. Dans `trainer.py::ppo_update` (moteur neuf) :

```
values_meta = policy.evaluate_actions_meta(...)      # ← lit actor_meta (réseau FANTÔME séparé)
loss_2 = BCE(values_meta, wager_objective)
...
if meta:
    loss_2.backward(retain_graph=True)               # ← gradient DANS actor_meta seulement
    actor_meta_optimizer.step()                      # ← met à jour le fantôme
total_loss = policy_loss - entropy*coef              # ← AUCUN terme wager
total_loss.backward(); actor_optimizer.step()        # ← le vrai acteur ne voit jamais le pari
```

- **M-C2** : `actor_meta` est un réseau distinct de l'acteur qui joue (`R_Actor`) ; le pari ne touche
  que lui.
- **M-C3** : côté critique, `evaluate_actions_meta` relit `actor_meta` → `critic_meta` ne reçoit
  jamais de gradient (poids mort, `critic_meta_optimizer.step()` = no-op).

⇒ Activer la métacognition (réglages 3-6) **ne change rien** au comportement vs 1-2. Le facteur
« méta » sort nul **par construction**.

---

## 2. Le principe de la correction (façon Blindsight/AGL)

Le juge (2ᵉ ordre + pari) lit les **représentations du VRAI acteur** (non détachées) et sa perte
BCE est **ajoutée à la perte de l'acteur** → un seul `backward` co-entraîne le CNN/GRU partagés.
**Monitoring uniquement** : le pari n'influence PAS la sélection d'action (`get_actions` reste
l'acteur baseline seul). On **supprime** les réseaux fantômes.

Structure du juge (déjà présente dans `SecondOrderNetwork`, réutilisée telle quelle) :
`comparison = features_initiales(pré-RNN) − features_cascadées(post-RNN)` de l'acteur →
`SecondOrderNetwork` (× cascade_2) → `wager (B,2)`.

---

## 3. Changements concrets (moteur neuf)

### 3.1 `policy.py::R_Actor` — ajouter le juge sur l'acteur qui joue
- Ajouter un sous-module `self.second_order = SecondOrderNetwork(hidden_size)` et un paramètre
  `cascade_iterations_2` (pour la cascade du 2ᵉ ordre).
- `evaluate_actions(...)` : capturer `initial = base(obs)` (pré-RNN) et `post = ` features après la
  cascade RNN existante ; calculer `comparison = initial − post` ; boucle `cascade_iterations_2` :
  `wager, prev = self.second_order(comparison, prev, 1/cascade_iterations_2)` ; **retourner
  `(action_log_probs, dist_entropy, wager)`** (au lieu de `(log_probs, entropy)`).
  ⇒ `second_order` étant un sous-module de `R_Actor`, `actor.parameters()` (et donc
  `actor_optimizer`) le couvre automatiquement — pas d'optimiseur séparé.
- **Architecture d'action INCHANGÉE** : pas de `layer_input` sur le chemin qui joue (quirk étudiant
  abandonné), pour que l'acteur soit identique méta-ON vs méta-OFF (sinon confond la comparaison).
  Le `RNNLayer` standard de `R_Actor` sert à produire `post` (pas besoin de `RNNLayer_Meta`).

### 3.2 `rmappo_policy.py::R_MAPPOPolicy` — supprimer les fantômes
- **Retirer** `actor_meta`, `critic_meta`, `actor_meta_optimizer`, `critic_meta_optimizer`,
  `evaluate_actions_meta`, `lr_decay_meta`.
- `evaluate_actions(...)` : propager le `wager` de l'acteur → retourner
  `(values, action_log_probs, dist_entropy, wager)`.
- `actor_optimizer = optim(self.actor.parameters())` couvre désormais `second_order`.
- Passer `cascade_iterations_2` à `R_Actor`.

### 3.3 `trainer.py::ppo_update` — co-entraînement + cible = avantage>0 (fix c) + device-aware
```
values, action_log_probs, dist_entropy, wager = policy.evaluate_actions(...)   # wager du VRAI acteur
...
policy_loss = policy_action_loss
actor_optimizer.zero_grad()
total_loss = policy_loss - dist_entropy*entropy_coef
if meta:
    wager_target = one_hot(adv_targ > 0)            # (B,2) — fix (c), dense/équilibré, in-update
    wager_loss   = BCE_with_logits(wager, wager_target) * wager_loss_coef
    total_loss   = total_loss + wager_loss          # ← UN seul backward co-entraîne acteur+2ᵉ ordre
total_loss.backward()
clip(actor.parameters()); actor_optimizer.step()
# critique : perte de valeur PURE, aucun terme méta (le papier ne parie que côté acteur)
```
- **Supprimer** tout le bloc `actor_meta`/`critic_meta`/`values_meta`/`loss_2`/`retain_graph` +
  le second `evaluate_actions_meta` côté critique.
- **`.cuda()` → `.to(**self.tpdv)`** (device-aware) : lève la contrainte GPU-only et rend
  l'entraînement testable sur CPU. (Changement d'infra nécessaire, pas une déviation scientifique.)
- `train_info` : ajouter `wager_loss` (→ `wager_loss_actor` dans metrics.json ; `wager_loss_critic`=0).

### 3.4 `runner` (run_marl_new.py / futur runner de prod) — simplifier
- La cible du pari est calculée **dans** `ppo_update` à partir de l'avantage → **retirer** le calcul
  EMA du runner (`compute_wager_objective`, `episode_wagers`, `wager_per_agent`, `ema_reward`).
  `train(buffer, meta=setting.meta)` suffit (plus de `wager_objective`).
- Conséquence : les hyperparams `maps.ema_alpha` / `maps.wager_condition` ne servent plus pour le
  run corrigé (c'était l'approche EMA du papier ; on la remplace par avantage>0).

### 3.5 `policy_meta.py` — retirer du chemin de prod
- `R_Actor_Meta`, `R_Critic_Meta`, `RNNLayer_Meta` : supprimés du chemin d'exécution. `SecondOrderNetwork`
  **conservé et réutilisé** par `R_Actor`. (Les classes fantômes peuvent rester dans le fichier
  marquées dépréciées, ou être retirées ; à trancher au moment du diff.)

---

## 4. Ce qui NE change PAS
- **Monitoring only** : `get_actions` (rollout) = acteur baseline seul ; le pari ne contrôle pas l'action.
- **Critique** : pas de pari côté critique (fidèle papier Fig.4 : pari uniquement côté acteur).
- **Réglages 1-2 (méta OFF)** : comportement identique (le juge n'est branché que si `meta=True`).
  ⇒ la parité baseline déjà prouvée (bit-exacte) reste valide.
- **Fix (b) time-major** : **NON retenu** — `n_rollout_threads=1` en prod rend l'ordre chunk/time sans
  objet (le `_cast` transpose déjà, et à N=1 c'est un no-op). À revisiter seulement si >1 thread.

---

## 5. Nouveaux hyperparamètres / déviations à logger (`docs/reproduction/deviations.md`)
- `wager_loss_coef` (nouveau ; défaut 1.0) — poids du terme de co-entraînement.
- Cible du pari = `one_hot(avantage>0)` au lieu de `reward>0`/EMA du papier — **déviation assumée** (fix c).
- Juge branché sur le vrai acteur + suppression fantômes — **déviation assumée** (fix a).
- `.cuda()` → device-aware — infra (pas une déviation numérique).

---

## 6. Tests de verrouillage (à écrire avec le fix)
1. **Le gradient du pari atteint l'acteur** : après un `ppo_update(meta=True)`, vérifier que
   `actor.base.*.grad` et `actor.rnn.*.grad` sont non-nuls (co-entraînement effectif). Test-clé du fix.
2. **Fantômes absents** : `R_MAPPOPolicy` n'a plus `actor_meta`/`critic_meta` ; `get_actions` inchangé.
3. **Cible = avantage>0** : shape `(B,2)`, one-hot, ~équilibrée après normalisation des avantages.
4. **Méta OFF inchangé** : parité baseline bit-exacte préservée (réutiliser `parity.py`).
5. **CPU-runnable** : `ppo_update` tourne sur CPU (après `.cuda()`→device) → 1 update finie sans erreur.
6. **1 update co-entraînée = bit-exact CPU vs GPU** (tolérance) : sanity device-agnosticité.

---

## 7. Ordre d'implémentation proposé
1. `.cuda()` → device-aware (débloque le test CPU). 2. `R_Actor` + `second_order` + wager dans
`evaluate_actions`. 3. `R_MAPPOPolicy` : supprimer fantômes, propager wager. 4. `ppo_update` :
co-entraînement + cible avantage>0. 5. runner : retirer plomberie EMA. 6. tests 1-6. 7. smoke GPU
1 cellule méta (réglage `maps`) → vérifier que méta-ON ≠ méta-OFF (le but !). 8. logger déviations.
