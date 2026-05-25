# Reverse-prompt — `src/maps/domains/blindsight/data.py`

**Module path actuel (sur main) :** `src/maps/experiments/blindsight/data.py`
(renommé `domains/blindsight/data.py` au Sprint 11+).
**Taille :** ~160 lignes.
**Paper :** §3.2 (Blindsight task design).
**Review existant :** `docs/reviews/blindsight-data.md` (Sprint-08 C.12).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `data.py` qui génère les patterns stimulus
> pour la tâche Blindsight (paper §3.2). Le module porte verbatim la
> fonction `generate_patterns` du `BLINDSIGHT/Blindsight_TMLR.py:259-328`
> du student. Bit-identical numerical behavior requis pour parity tests.
>
> **Conditions de stimulus (paper §3.2)** :
> - `SUPERTHRESHOLD` (0) : stimulus clean, no noise — patient sees clearly
> - `SUBTHRESHOLD` (1) : stimulus below detection threshold + noise baseline
> - `LOW_VISION` (2) : stimulus scaled down + noise baseline
>
> Chaque batch contient `patterns_number` trials, split moitié-moitié :
> - **Moitié 1** : noise-only trial. Pattern = `multiplier · U(0,
>   random_limit, num_units) + baseline`. `order_2_target = [0, 1]`
>   (low wager, no stimulus).
> - **Moitié 2** : stimulus-present trial. Pattern noise + injecté à
>   `stim_idx` (random.randint). Si l'amplitude injectée `>=
>   multiplier/2` → "above threshold" → `[1, 0]` (high wager) +
>   `stim_present[stim_idx] = 1`. Sinon "below threshold" → `[0, 1]`
>   (low wager) + `stim_present = 0`.
>
> **Important** : la cible `order_2_target` suit la règle "détecté ou
> pas" (amplitude >= multiplier/2), **pas** la ground-truth presence
> label. C'est ce qui crée l'effet "blindsight" — quand on est juste
> sous le seuil, on a un stimulus présent mais non-détecté.
>
> Output `TrainingBatch` :
> - `patterns: (N, num_units)` float32
> - `stim_present: (N, num_units)` one-hot (zeros pour subthreshold)
> - `order_2_target: (N, 2)` softmax-style high/low wager
>
> Tous les tensors sur le device demandé, `.requires_grad_(True)`
> appelé sur les 3 (préservation parity student — seul `patterns` flow
> vraiment dans autograd).
>
> **RNG** : `np.random` global API pour les uniform draws (parity
> student `np.random.uniform`). `random.randint` pour `stim_idx`
> (parity student `random.randint`). Les 2 doivent être seedés par
> `set_all_seeds`.

## 2. Contraintes scientifiques

### Paper §3.2 — Blindsight task design

La tâche Blindsight simule le phénomène neurologique (Weiskrantz 1986)
où des patients avec cortex visuel lésé peuvent "deviner" la présence
d'un stimulus sans en avoir conscience. Modèle MAPS :
- **First-order** : encoder-decoder qui apprend à reconstruire le pattern
- **Second-order** : wager confidence sur la présence du stimulus
- **Effet attendu** : le wager peut chuter en `SUBTHRESHOLD` même
  quand la discrimination réussit (= "blindsight")

### `multiplier/2` threshold

Le seuil est `params.multiplier / 2`. Pourquoi ? Pas explicité dans le
paper. Possiblement le median de `U(0, multiplier)` (= multiplier/2),
ce qui donne ~50% des stims présents au-dessus du seuil et 50%
en-dessous. Crée un mix balancé pour entraîner le wager.

### Conditions hardcodées vs config

`ConditionParams` (random_limit, baseline, multiplier) sont **chargées
depuis `config/env/blindsight.yaml`**, pas hardcodées dans ce module.
C'est le `BlindsightTrainer` qui fait le wiring. Bonne séparation.

## 3. Contraintes d'ingénierie

### Parité bit-exact avec student

Le porting préserve :
- L'ordre exact des appels RNG (`np_rng.uniform` puis `random.randint`)
- L'utilisation de `np.random` global (legacy API) — pas `default_rng`
- L'appel `requires_grad_(True)` sur les 3 tensors (legacy student
  pattern, harmless mais préservé)

### `np_rng = rng or np.random`

Pattern fallback : si l'appelant passe un `Generator` explicite,
utilisé ; sinon fallback global state. Permet (a) parity bit-exact
quand caller ne passe rien (matches student), (b) injection pour
tests.

### Mémoire intermédiaire

`patterns: list[np.ndarray]` accumulé via append puis `np.asarray` →
`torch.tensor`. C'est inefficient (3 copies) mais matches student.
Optim possible : préallouer numpy array + remplir. Gain négligeable
(N=200 typiquement).

## 4. Deviations / dettes présentes

### Pas de déviation paper sur ce module

Le module est un port fidèle bit-exact. Toutes les valeurs (random_limit,
baseline, multiplier) viennent de la config YAML qui matche le student.

### `requires_grad_(True)` sur stim_present + order_2_target

C'est du dead code (ces tensors ne flow pas dans autograd) mais
préservé pour parity student (qui le faisait). Pas un bug, juste du
bruit.

### Note Sprint-08 D.23

Sprint-08 D.23 a audité ce module et confirmé bit-exact match avec le
student `Blindsight_TMLR.py:259-328`. Pas de finding open.

## 5. Questions ouvertes

- **Q1 :** Pourquoi `multiplier/2` comme threshold ? Convention paper,
  ou choix empirique non-documenté ? Test : varier le threshold et
  voir l'impact sur le wager curve.
- **Q2 :** Le mix RNG (np.random global + Python random) est fragile.
  Si on migre vers `default_rng()` pour python random, c'est
  potentiellement un refactor cascade. Vaut-il la peine ?
- **Q3 :** *(à toi)* — qu'est-ce qui te paraît surprenant ?

## 6. Pour le rewrite (Sprint 11+)

### À garder
- `StimulusCondition` IntEnum + `ConditionParams` frozen dataclass
- `TrainingBatch` dataclass
- `generate_patterns()` signature et logique
- Threshold `multiplier/2` (paper-faithful)
- RNG mix (`np.random` + `random.randint`) — parity student

### À améliorer
- **Préallouer numpy arrays** au lieu de list.append — perf marginal
  mais cleaner
- **Type hint `device`** : `torch.device | str` est OK, on pourrait
  raffiner à `Literal["cpu", "cuda", "mps", "auto"] | torch.device`
- **Retirer `.requires_grad_(True)` sur stim_present + order_2_target** :
  dead code que paper-faithful préserve. Question : casse-t-on parity
  test si on enlève ? À vérifier — si non, simplifier.
- **Docstring** : être plus explicite sur le "blindsight effect" et
  pourquoi `multiplier/2` est le seuil clé.

### Tests à écrire
- Bit-exact parity vs `Blindsight_TMLR.py:259-328` (seed fixé, mêmes
  conditions → mêmes tensors)
- Distribution check : ~50% high_wager / ~50% low_wager sur les
  stimulus-present trials (seuil = median)
- Shape : `(N, 100)` patterns, `(N, 100)` stim_present, `(N, 2)`
  order_2_target
- Stimulus-only half a bien `argmax(pattern)` au `stim_idx` choisi

### Opportunités perf
- Vectoriser la génération : au lieu de la boucle `for i in range(n)`,
  générer tout en batch (numpy ops sur (n, num_units) directement).
  Gain probablement significatif pour grands N mais casse parity (ordre
  RNG différent).

### Connexion
- Utilisé par `domains/blindsight/trainer.py`
- Configure via `config/env/blindsight.yaml`
- Seedé via `utils/seeding.set_all_seeds`
