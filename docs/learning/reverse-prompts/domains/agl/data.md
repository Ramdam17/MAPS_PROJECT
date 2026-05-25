# Reverse-prompt — `src/maps/domains/agl/data.py`

**Module path actuel (sur main) :** `src/maps/experiments/agl/data.py`
**Taille :** ~335 lignes.
**Paper :** §4 (AGL task), Dienes (1997) §5 (grammar design).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `data.py` qui génère les patterns de mots
> pour l'Artificial Grammar Learning (AGL) task. Le module porte
> verbatim les fonctions `Generate_Word_Random`, `Generate_Grammar_A`,
> `Generate_Grammar_B`, `encode_word`, `Array_Words`, `target_second`
> du `AGL/AGL_TMLR.py:268-441` student. Bit-identical pour parity tests.
>
> **Design Dienes (1997) §5** :
> - Alphabet 5 lettres : `{x, v, m, t, r}`
> - 2 finite-state machines (Grammar A et B) générant des mots
>   "grammaticaux"
> - Longueur de mot uniforme dans [3, 8], padding zero à droite jusqu'à 8
> - Grammar B = control / distractor (sémantique différente de A)
>
> **Encoding** :
> - Chaque lettre → 6-bit one-hot (6ème bit toujours 0, legacy 6-letter
>   alphabet)
> - 8 lettres max × 6 bits = **48-d input**
> - `NUM_INPUT_UNITS = 48`, `BITS_PER_LETTER = 6`
>
> **API publique** :
> - `generate_random_word()` : random 5-letter mix, pour pretrain
> - `generate_grammar_a()` : walk Grammar A FSM (5 nodes + terminal 6)
> - `generate_grammar_b()` : walk Grammar B FSM (5 nodes, structure
>   différente)
> - `encode_word(word)` → list[int] de 48 bits
> - `generate_batch(grammar_type, number, device) → TrainingBatch`
> - `target_second(input_, output)` → tensor (N,) en {0, 1} :
>   target binaire pour le wager.
>
> **`target_second` logique** :
> - Compte `k = nombre de 1s dans input row`
> - Prend les top-k positions de l'output
> - Target = 1.0 si `set(top-k output) == set(positions à 1 input)`,
>   sinon 0.0
> - C'est l'analog AGL de la détection Blindsight : "high wager" = le
>   1st-order a parfaitement reconstruit les positions actives.
>
> **`GrammarType` IntEnum** : RANDOM=1, A=2, B=3. Matches student
> `grammar_type` arg int.
>
> Toutes les RNG via `random.randint/choice` Python global API
> (parity student).

## 2. Contraintes scientifiques

### Dienes (1997) FSM

Les 2 grammars sont des **finite-state machines** simples (5 nodes
chacune). Grammar A et Grammar B partagent le même alphabet mais des
transitions différentes — c'est ce qui permet de tester "transfer of
implicit knowledge" : un sujet entraîné sur A peut-il distinguer A vs
B sans formaliser la règle ?

Détails verbatim dans le code (state-transition tables L278-311 pour A,
L315-351 pour B). On les conserve bit-exact malgré l'aspect "walrus-
ladder" (chaîne de `if (position, current_path) == (X, Y)`).

### 6-bit one-hot avec 6ème bit unused

Legacy artifact d'une version qui avait un 6-letter alphabet. On
garde le chunk 6-bit pour compatibilité (le decoder
`make_chunked_sigmoid(6)` traite chaque chunk de 6 bits indépendamment
— cf. `first_order_mlp.md`).

### `target_second` topologie

Le wager target en AGL n'est PAS "stimulus présent ou absent" (comme
Blindsight). C'est "le 1st-order a-t-il reconstruit parfaitement les
positions actives ?". Plus binaire, plus sévère.

Conséquence : sur les trials où le 1st-order rate la reconstruction
(early training), le wager target est mass 0. Sur les trials où il
réussit (late training), wager target = 1. **Le wager apprend à
prédire le success du 1st-order**, pas un attribut intrinsèque du
stimulus.

## 3. Contraintes d'ingénierie

### FSM verbatim student

Le code FSM `Generate_Grammar_A/B` est dense et peu lisible (state
transitions inline). Le port le préserve byte-for-byte pour parity.
Une refacto plus lisible (table de transitions, dict-based) casserait
parity bit-exact (ordre des draws RNG différent).

### `encode_word` defensive

`mapping.get(letter, [0]*6)` : si une lettre inconnue apparaît
(impossible normalement mais défensif), encode all-zeros plutôt que
crasher. Préservé du student.

### `target_second` pourrait être vectorisé

La boucle `for i in range(num_rows)` + `set()` comparison est lente
sur GPU. Vectorisation possible :
```python
k_per_row = input_.sum(dim=1).long()
# top-k variable par row : pas trivial à vectoriser
```
La forme actuelle est paper-faithful via student. Vectorisation = perf
gain mais casse parity test exacte (ordering différent).

## 4. Deviations / dettes présentes

### Pas de déviation paper sur ce module

Tous les paramètres (BITS_PER_LETTER=6, NUM_INPUT_UNITS=48, alphabet,
grammars FSM) matchent paper + student. Port bit-exact.

### D-004 — Chunked sigmoid

Pas dans ce module mais related : le decoder applique sigmoid par
chunk de 6 bits, pas globalement. Voir `networks/first_order_mlp.md`.

### Pas de DETTE active

Module propre, bien typé, bien testé.

## 5. Questions ouvertes

- **Q1 :** Pourquoi 5 lettres ? Dienes (1997) a probablement testé
  d'autres tailles. Aurait-on de meilleurs résultats avec 4 ou 6 ?
  Hors scope reproduction mais intéressant.
- **Q2 :** Le 6ème bit unused est un déchet computationnel. Vaut-il la
  peine de le supprimer (passer à 5-bit one-hot, total 40-d) ? Casse
  parity. Pas prioritaire.
- **Q3 :** `target_second` est sévère (exact set match). Une version
  "soft" (Jaccard similarity > seuil) pourrait être plus informative
  pour l'apprentissage. Question pour Sprint 11+ et au-delà.
- **Q4 :** *(à toi)* — qu'est-ce qui te paraît surprenant dans l'AGL
  task design ?

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Constants `BITS_PER_LETTER=6`, `NUM_INPUT_UNITS=48`, alphabet
- FSM verbatim Grammar A + B (parity)
- `GrammarType` IntEnum
- `target_second` exact-set-match logic
- Python `random` global API (parity)

### À améliorer (sans casser parity)
- **Docstring `target_second`** : expliquer plus clairement la
  sémantique "wager = 1st-order reconstructed perfectly"
- **Tests bit-exact** : seed fixé → mêmes mots, mêmes encodings
- **Refactor FSM optionnel** : exposer les transition tables comme
  data structures lisibles, garder les `_Generate_*_legacy` fonctions
  comme fallback parity. Bénéfice : lisibilité. Coût : 2x maintenance.

### Tests à écrire
- Bit-exact parity `Generate_Grammar_A` vs reference
- Mêmes pour B, random, encode_word
- `target_second` : edge cases (all-zero input, all-ones, k=0, k=8)
- Distribution check : Grammar A produit certaines suites avec
  fréquences connues (Dienes data)

### Opportunités perf
- `target_second` vectorisable mais casse parity exacte
- `generate_batch` génère N mots séquentiellement Python — pas
  parallélisable trivialement (Python random global state)

### Connexion
- Utilisé par `domains/agl/trainer.py`, `domains/agl/pool.py`
- Pas de config YAML dédiée (constants in-module)
