# Walkthrough — La cascade de McClelland, du paper au code

**Public visé :** quelqu'un qui connaît PyTorch et la ML générale, mais
pas MAPS. Lu après le Sprint 11 (commits `c498bb3` à `8d6d926` sur
`refactor/core`).

**Pré-requis recommandés :** lire `docs/learning/reverse-prompts/core/cascade.md`
en parallèle pour le détail technique. Ce walkthrough est *narratif* —
il explique le *pourquoi* derrière les décisions de design, pas le
*quoi*.

---

## 1. Pourquoi ce document existe

MAPS (Vargas et al., TMLR submission) reprend deux idées
neuroscientifiques anciennes pour les recoller à un réseau moderne :

1. **La cascade de McClelland (1979, 1989)** — accumulation graduelle
   d'évidence dans une activation neuronale, paramétrée par un taux α.
2. **Le second-order de Pasquali & Cleeremans (2010)** — un réseau
   métacognitif qui *observe* le first-order et produit un "wager"
   (confiance dans la décision).

Lus séparément, ces deux mécanismes sont chacun simples. **Lus
ensemble dans MAPS, ils interagissent d'une manière que la lecture
ligne-à-ligne du code ne révèle pas.** Ce walkthrough déroule cette
interaction pour qu'un lecteur (toi, dans 6 mois ; un nouveau membre
du lab ; un reviewer TMLR) puisse comprendre pourquoi `core/` est
conçu comme un bloc et pas comme 3 fichiers indépendants.

## 2. L'équation 6 du paper

Le paper §2.1 eq.6 :

$$a_{ir}(t) = \alpha \cdot \sum_j w_{ij} \cdot a_{js}(t) + (1-\alpha) \cdot a_{ir}(t-1)$$

Le terme α·Σ_j w_{ij}·a_{js}(t) est juste « la sortie d'une couche
linéaire à l'instant t » — rien de spécial à la cascade. Le terme
intéressant est `(1-α)·a(t-1)` : on garde une fraction de
l'activation précédente.

Donc dans le code, on factorise :

```python
def cascade_update(new, prev, cascade_rate):
    if prev is None:
        return new                       # bootstrap t=0
    return cascade_rate * new + (1.0 - cascade_rate) * prev
```

Ce sont **3 lignes de PyTorch** qui résument 40 ans de recherche
neuroscientifique. C'est facile à coder ; c'est plus dur à
comprendre.

## 3. Le piège des deux α

Première chose qu'on apprend en lisant le code : **il y a deux α
dans MAPS, et ils n'ont rien à voir.**

| Symbole | Valeur | Endroit | Origine |
|---------|--------|---------|---------|
| `cascade_rate` | 0.02 | `core.cascade.cascade_update` | McClelland 1989, paper eq.6 |
| `ema_alpha` | 0.45 | wagering EMA (à venir Sprint 12) | Koch & Preuschoff 2007, paper eq.13 |

Le student code utilisait `alpha` pour les deux. Sprint 11 a renommé
le cascade en `cascade_rate` (décision D11.1) : la signature publique
oblige le lecteur à voir la distinction. C'est un coût d'1 mot — ça
prévient un bug par session.

## 4. Le finding qui change tout — la cascade no-op

Voici ce qu'on découvre en regardant `cascade_update` sérieusement :

**Si `new` est constant à travers les itérations, 50 applications
produisent exactement la même valeur qu'une seule.**

Démonstration : `cascade_update(new, None, α) = new` (bootstrap).
Puis `cascade_update(new, new, α) = α·new + (1-α)·new = new`. Point
fixe instantané.

Plus formellement : la suite récurrente `a(t+1) = α·new + (1-α)·a(t)`
converge géométriquement vers `new` au taux `(1-α)^t`. Avec
`new = a(0) = new`, on est déjà au point fixe, donc on y reste.

**Conséquence pour SARL :** le Q-network de SARL n'a pas de dropout
dans son forward. Donc `new_activation` ne varie pas entre
itérations. Donc la cascade est mathématiquement no-op sur ce
chemin. Donc paper Table 6 Setting 2 (cascade 1st-order only) devrait
être numériquement identique à Setting 1 (no cascade), à RNG près.

Pourtant le paper montre `z = −2.59` sur Seaquest Setting 2.
Trois hypothèses :
1. **Bruit RNG** (N = 3 seeds, distribution lourde sur MinAtar)
2. **Side-effect non identifié** (buffer ? scheduler ?)
3. **Paper bruité**

Sprint-08 D-sarl-cascade-noop a tranché : garder paper-faithful +
warning. Ce walkthrough la **transforme en test** : `test_cascade.py::
test_deterministic_noop_after_n_iterations`. Si un jour quelqu'un
modifie `cascade_update` et casse cette équivalence analytique, le
test le rattrape.

## 5. Pourquoi la cascade marche quand même

Si la cascade est no-op sur chemin déterministe, **pourquoi MAPS
l'utilise** sur Blindsight, AGL, et le SecondOrder de SARL ?

Parce que dans ces 3 cas, **il y a du dropout entre la cascade et
le forward.** Et le dropout, c'est de la stochasticité.

```
forward_step(t):
    h_t = encoder(input)                  # déterministe
    h_drop_t = Dropout(h_t)               # ← masque aléatoire DIFFÉRENT à chaque t
    h_cascade_t = cascade_update(h_drop_t, h_cascade_{t-1}, 0.02)
    out_t = decoder(h_cascade_t)
```

À chaque itération `t`, `Dropout(h_t)` produit un masque différent.
Donc `new_activation` *varie* d'une itération à l'autre. Donc la
cascade fait quelque chose. Plus précisément :

$$h_{cascade}(t) = \sum_{k=1}^{t} \alpha \cdot (1-\alpha)^{t-k} \cdot \text{Dropout}_k(h)$$

Après ~50 itérations, c'est un moyennage exponentiel pondéré de 50
masks dropout indépendants. **C'est mathématiquement équivalent à
un Monte-Carlo dropout** (Gal & Ghahramani 2016) — une technique
classique pour estimer l'incertitude bayésienne d'un réseau.

Donc : la cascade dans MAPS, **c'est en fait du MC-dropout déguisé**.
Le paper n'utilise jamais ce framing — mais c'est lui qui rend le
mécanisme pertinent. Cette équivalence est *la* clé pédagogique de
tout MAPS. Sans elle, le lecteur regarde 50 itérations de math
identique et se demande pourquoi.

## 6. Pourquoi `core/` est un bloc co-conçu

Étant donné les sections 4 et 5, les 3 modules de `core/` ne sont
pas indépendants. Ils encodent un **contrat** :

1. **`cascade.py`** définit la dynamique d'accumulation.
2. **`second_order.py`** est *le seul* site où la cascade voit du
   dropout — donc *le seul* site où la cascade est non-triviale dans
   MAPS BS/AGL.
3. **`losses.py:cae_loss`** consomme le gradient qui flow à travers
   les 50 itérations de cascade (le `retain_graph=True` du trainer).

Modifier l'un sans considérer les deux autres casse le contrat. Par
exemple :
- Si on "optimise" `cascade_update` pour skipper la boucle sur path
  déterministe (ce qui est une optim correcte), il faut s'assurer
  qu'on ne le fait *pas* sur le path BS/AGL où le dropout doit fire
  50 fois.
- Si on change `SecondOrderNetwork` pour détacher `prev_comparison`
  après N steps (pour borner la mémoire), il faut documenter que la
  cascade ne s'étend plus à travers tout le graph — ce qui change
  le gradient que `cae_loss` reçoit.

Sprint 11 réécrit les 3 ensemble pour qu'ils restent cohérents par
construction. Sprint 12+ peut les utiliser, mais pas les modifier
sans relire ce walkthrough.

## 7. Comment les tests reflètent ce contrat

La suite de tests Sprint 11 (67 passing) a 3 niveaux qui
correspondent à 3 niveaux de garantie :

- **Math sanity** (`tests/unit/core/`, ~53 tests) — chaque fonction
  ou classe est testée en isolation avec des tensors jouets. Aucune
  dépendance externe. Si un test math casse, c'est un bug
  d'implémentation.

- **Parity vs paper_reference** (`tests/parity/core/`, 5 tests) —
  on compare l'output bit-pour-bit à un port verbatim du code Vargas
  original (`external/paper_reference/blindsight_tmlr.py`). Le code
  Vargas a produit les Tables 5/6/7 du paper, donc tout drift ici =
  drift par rapport au paper. Tolérance 1e-7 (eval) et 1e-5 (train,
  pour absorber le bruit de quantization float32 sur 50 unrolls).

- **Integration** (à venir Sprint 12) — un forward+backward complet
  d'entraînement Blindsight, comparé aux z-scores paper. C'est là
  qu'on validera scientifiquement le port.

Si Sprint 12 casse un test parity, c'est *Sprint 12 qui a un bug*,
pas le test. Les tests Sprint 11 sont les gardiens de l'invariant
mathématique.

## 8. Ce qui n'est pas dans `core/`

Pour la mémoire de qui lit dans 6 mois — `core/` ne contient PAS :

- `FirstOrderMLP` (encoder + decoder) — vit dans `networks/`
  (Sprint 12)
- Les trainers (boucles `for epoch in range(...)`) — dans
  `domains/<x>/trainer.py` (Sprint 12+)
- Le data loading — dans `domains/<x>/data.py`
- Les CLI — dans `domains/<x>/cli.py`
- `weight_regularization` est dans `core/losses.py` mais n'est
  utilisé que par SARL+CL (Sprint 15) — gardé dans `core/` parce
  que c'est de la math pure, mais hors-chemin pour les autres
  domaines.

`core/` est petit (3 fichiers, ~700 LOC) et c'est volontaire. Toute
addition future doit passer le test "est-ce que TOUS les domaines
en auraient besoin ?" — sinon, ça part dans `domains/<x>/`.

---

## 9. Pour aller plus loin

- **`docs/learning/reverse-prompts/core/cascade.md`** — détails
  techniques module-par-module
- **`docs/reproduction/deviations.md`** — registre des écarts
  paper↔code (D-001, D-002, D-sarl-cascade-noop, D.25)
- **`docs/learning/reverse-prompts/core/second_order.md`** — la
  composition Comparator → Dropout → cascade → Wager
- **Paper Vargas et al. 2025 (TMLR)** — §2.1 (architecture), §2.2
  (training)
- **McClelland (1989)** — "Parallel distributed processing
  implications for cognition and development" — l'origine de la
  cascade
- **Gal & Ghahramani (2016)** — "Dropout as a Bayesian
  approximation" — la clé pour comprendre pourquoi cascade+dropout
  marche
