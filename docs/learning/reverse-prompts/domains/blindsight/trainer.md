# Reverse-prompt — `src/maps/domains/blindsight/trainer.py`

**Module path actuel (sur main) :** `src/maps/experiments/blindsight/trainer.py`
(renommé `domains/blindsight/trainer.py` au Sprint 11+).
**Taille :** 522 lignes — le plus gros module Blindsight.
**Paper :** §3 (Blindsight expérience), Table 5a (résultats), Table 9
(architecture).
**Review existant :** `docs/reviews/blindsight-trainer.md`,
`docs/reviews/blindsight-eval-metric-audit.md`,
`docs/reviews/rg002-wager-gap-investigation.md`.

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `trainer.py` qui orchestre l'entraînement et
> l'évaluation de la tâche Blindsight. Le module porte la logique
> `prepare_pre_training` + `pre_train` + `testing` du
> `BLINDSIGHT/Blindsight_TMLR.py:416-630` du student, dans une classe
> config-driven.
>
> **Structures de données :**
>
> ```python
> @dataclass(frozen=True)
> class BlindsightSetting:
>     id: str
>     label: str
>     cascade_1st: bool   # cascade applied to 1st-order encoder
>     cascade_2nd: bool   # cascade applied to 2nd-order wager path
>     second_order: bool  # whether to train the 2nd-order network
>
>     @property
>     def cascade(self) -> bool:  # legacy alias
>         return self.cascade_1st or self.cascade_2nd
>
>     @classmethod
>     def from_dict(cls, d) -> BlindsightSetting:
>         # Support both 6-cell schema (cascade_1st + cascade_2nd) and
>         # legacy 2x2 schema (cascade) for backward compat.
> ```
>
> **6 settings du paper Table 5** :
> 1. Baseline (no cascade, no 2nd-order)
> 2. Cascade 1st only
> 3. 2nd-order only
> 4. **MAPS** (cascade 1st + 2nd-order) — headline
> 5. Cascade 2nd only
> 6. Full MAPS (cascade both)
>
> **Cascade rate asymmetry** : chaque côté a son propre flag → son
> propre `cascade_rate` (α si on, 1.0 si off) et `cascade_iterations`
> (50 si on, 1 si off). Critique : avant Sprint-08 D.31, le trainer
> appliquait la cascade *symétriquement* → Setting 4 produisait en
> réalité Setting 6 numbers ! Bug fixed D.31.
>
> **Class `BlindsightTrainer`** : 1 par (seed, setting) combination.
> Méthodes :
> - `.build()` : instancie `FirstOrderMLP` + `SecondOrderNetwork`,
>   optimizers (default ADAMAX, lr depuis config), schedulers (StepLR
>   step_size=25 gamma=0.98)
> - `.pre_train(n_epochs=None, condition=None, batches=None)` : boucle
>   d'entraînement par epoch
> - `.evaluate(eval_patterns_number=None, conditions=(...))` : métriques
>   par condition (SUPERTHRESHOLD / SUBTHRESHOLD / LOW_VISION)
>
> **Pre-training loop par epoch (paper §3 + reference verbatim)** :
> 1. Générer un batch (default: SUPERTHRESHOLD condition)
> 2. Cascade 1st-order : `cascade_iters_1` itérations de `first_order(x,
>    prev_h1, prev_h2, cascade_rate_1)`. `prev_h1` accepté mais
>    no-op (cf. `first_order_mlp.md`).
> 3. **`optim_1.zero_grad()` AVANT le 2nd-order backward** — important :
>    `loss_2.backward(retain_graph=True)` accumule gradients dans le
>    1st-order, qui seront augmentés par `loss_1.backward` avant
>    `optim_1.step()`.
> 4. Si `setting.second_order` : cascade 2nd-order pour `cascade_iters_2`
>    itérations, compute BCE wager loss (eq.5), `optim_2.zero_grad()`,
>    `loss_2.backward(retain_graph=True)`, `optim_2.step()`,
>    `sched_2.step()`.
> 5. Sinon : throwaway no-grad forward (préserve RNG consumption).
> 6. CAE loss (eq.4 stub→Rifai) : `cae_loss(W=fc1.weight, x=stim_present,
>    recons_x=h2, h=h1, lam=1e-4, recon="bce_sum")`,
>    `loss_1.backward(retain_graph=True)`, `optim_1.step()`,
>    `sched_1.step()`.
>
> **Eval (`testing()` port)** :
> - Pour chaque condition, génère 200 trials (100 noise + 100 stim)
> - `discrimination_accuracy = mean(argmax(h2[stim_half]) ==
>   argmax(input[stim_half]))` — compare à argmax de l'**input** (le
>   stimulus signal), pas à `stim_present` (qui peut être all-zero
>   pour subthreshold). Important pour reproduire Tables.
> - `wager_accuracy` = `(high_w > threshold) == (target > threshold)`
>   où threshold = 0.5 (super/sub) ou 0.15 (low_vision).
> - Si `n_wager_units=2` : sigmoid les raw logits à l'inférence (paper
>   eq.5 per-unit).
>
> **Wiring config** :
> - cfg `cascade.alpha / cascade.n_iterations` (depuis maps.yaml)
> - cfg `first_order.input_dim, hidden_dim, encoder_dropout,
>   weight_init_range` (Blindsight override input=100, hidden=40)
> - cfg `second_order.input_dim, dropout, n_wager_units, hidden_dim`
>   (Blindsight hidden=100 Pasquali restored D.25)
> - cfg `optimizer.name, lr_first_order, lr_second_order`
> - cfg `scheduler.step_size, gamma`
> - cfg `train.n_epochs, batch_size, data_factor, noise_level`
> - env_cfg `conditions.{super,sub,low}, eval.patterns_number,
>   eval.wager_thresholds`
>
> **Fail-fast guard SimCLR (D.22b)** : avant build, check
> `cfg.first_order_loss.kind == "cae"` ; si "simclr" raise
> NotImplementedError. Empêche les runs accidentels sur la branche non
> portée.

## 2. Contraintes scientifiques

### Settings 4 vs 6 = piège asymétrique cascade

Avant Sprint-08 D.31 : le trainer appliquait `cascade_rate` symétriquement
aux deux networks. La cellule "both" du factorial 2x2 produisait donc
**paper Setting 6** (cascade des deux côtés) sous le label "MAPS"
(paper Setting 4 = cascade 1st-order only). Numériquement c'est très
proche sur Table 5a mais c'est conceptuellement faux.

**Fix D.31** : chaque côté a son flag (`cascade_1st`, `cascade_2nd`).
Factorial passé à 6 cells. Les 500-seed runs ont confirmé que les 12
métriques (4 settings × 2 tiers × prec/wager) tombent dans ±2σ paper.

### RG-002 H1 — hidden_dim=40 vs paper Table 9

Paper Table 9 dit `hidden=60`. Student code passe `hidden=40` literal.
Notre config Blindsight utilisait `hidden=100`. Sprint-08 D.25 a tranché :
**`hidden=40` aligné avec le code student** (qui a produit les Tables).
Paper Table 9 est inconsistant avec le code.

### RG-002 H10 — Pasquali hidden restored

Paper §2.2 dit "as in Pasquali & Cleeremans (2010)" → implique hidden
layer dans le wager head. Student code passe `hidden_2nd=100` au
`SecondOrderNetwork.__init__` mais **ne l'utilise jamais** (bug). Sans
hidden, wager plafonne à 0.67. Avec hidden=100 restauré, wager → 0.82.

### Gradient flow weird (paper-faithful)

`optim_1.zero_grad()` AVANT `loss_2.backward()`. Pourquoi : le 2nd-order
loss flow dans le 1st-order via le `ComparatorMatrix` (qui propage
gradient vers `first_order_output`). Donc `loss_2.backward()` accumule
des gradients dans `fc1.weight`. Puis `loss_1.backward()` ajoute plus
de gradient. `optim_1.step()` consomme la somme.

C'est ce qui fait que les 2 networks **s'entraînent conjointement** dans
les settings avec `second_order=True`. C'est paper-faithful via student.

### Eval metric subtilité

Le student compare `argmax(h2)` à `argmax(input)`, **pas** à
`argmax(stim_present)`. Pourquoi : sur les trials subthreshold,
`stim_present` peut être all-zero (amplitude < seuil) mais le pattern
input porte quand même le signal à `stim_idx`. Comparer à input
récupère ce signal de discrimination. Si on compare à stim_present, on
perd les trials subthreshold qui sont l'essence de la tâche.

## 3. Contraintes d'ingénierie

### Config-driven complet

Aucune magic number dans ce trainer. Tout vient de `cfg` / `env_cfg`.
Le seul thing hardcoded : la liste des optimizers `_OPTIMIZERS` (ADAM,
ADAMW, ADAMAX, SGD, RMS, RMSPROP), mappée à torch.optim classes.

### `env_cfg` injection pour tests

`env/blindsight.yaml` utilise `${train.noise_level}` interpolation.
Trick : merger `env_cfg` SOUS `cfg` (qui a un `train.noise_level`),
puis `OmegaConf.resolve(merged)`. Le merged a tout résolu. Permet aux
tests d'injecter un `env_cfg` pré-résolu directement.

### Dependencies declaration explicit (`assert ... is not None`)

`pre_train()` et `evaluate()` font des assertions sur `self.first_order
is not None` etc. avant utilisation. Évite les NoneType errors si
caller oublie `.build()`. Petit overhead, gros gain DX.

### Throwaway forward pour préserver RNG

Quand `setting.second_order=False`, on fait quand même un
`with torch.no_grad(): self.second_order(...)` pour préserver le
nombre de tirages RNG (dropout interne consomme RNG même en no_grad).
Crucial pour parity bit-exact entre settings.

### `pre_train()` accepte `batches` injecté

Mode test : on peut passer une `list[TrainingBatch]` pré-générée pour
éliminer les race conditions RNG entre 2 trainers comparés. Utile
pour `tests/parity/blindsight/test_blindsight_pretrain.py`.

## 4. Deviations / dettes présentes

### 🚨 D-blindsight-hidden-40 (D.25, resolved)

Voir §2 RG-002 H1. hidden_dim=40 aligné avec student code, override
`-o first_order.hidden_dim=60` pour reproduire paper Table 9 literal.

### 🚨 D-blindsight-wager-hidden (D.25, resolved)

Voir §2 RG-002 H10. Pasquali hidden restauré, `hidden_dim=100` pour
Blindsight (`hidden_dim=0` pour student code path).

### ⚠️ D-blindsight-metric-mismatch (D.30 audited)

Sprint-08 D.30 a audité que la métrique `discrimination_accuracy` est
bit-exact paper-faithful (compare à argmax(input), pas argmax(stim_present)).
Le résidu de gap RG-002 n'est PAS une métrique artifact.

### D-blindsight-seeds

Paper utilise 500 seeds. Config par défaut était 5-10. Phase F utilise
500 via sbatch array.

### D-blindsight-temperature, epochs, dropout-rate

Tous audités D.23, alignés. Voir deviations.md B.9.

### Pas de DETTE active sur ce module

Le trainer est gros (522 L) mais propre. Pas de magic, pas de duplication.

## 5. Questions ouvertes

- **Q1 :** Le `optim_1.zero_grad()` avant `loss_2.backward()` est
  brillant en parity mais piège pour quelqu'un qui ré-écrit. Doit-on le
  faire ressortir dans une docstring (warning "do not change order
  without understanding gradient flow") ? Ou est-ce trop fragile par
  design ?

- **Q2 :** Le throwaway forward (`no_grad`) consomme du RNG pour
  préserver la parité. C'est élégant mais ça rend les runs sans
  2nd-order ~2× plus lents que nécessaire. Optimisable si on accepte
  de casser parity exacte sur les seeds.

- **Q3 :** Le DataLoader pattern n'est pas utilisé — chaque epoch
  régénère son batch à la volée. Pourquoi pas un pre-generated dataset ?
  Pour pouvoir varier le `noise_level` dynamiquement entre runs ?
  Ou simplement parity student qui le fait pareil ?

- **Q4 :** *(à toi)* — qu'est-ce qui te paraît surprenant ?

## 6. Pour le rewrite (Sprint 11+)

### À garder

- `BlindsightSetting` frozen dataclass + 6-cell schema + legacy 2x2 alias
- Cascade asymétrique (`cascade_1st`, `cascade_2nd`)
- Gradient flow weird (zero_grad order) — paper-faithful, ne pas toucher
- Throwaway forward avec no_grad pour préserver RNG
- Eval metric : `argmax(h2)` vs `argmax(input)`
- Threshold per condition (0.5 super/sub, 0.15 low_vision)
- Config-driven complet

### À améliorer

- **Documenter le gradient flow** : ajouter un commentaire explicite
  avant le `optim_1.zero_grad()` qui dit "DO NOT move — gradient flow
  is paper-faithful, depends on this ordering"
- **Simplifier `_load_env_cfg`** : actuellement OmegaConf.merge +
  resolve. Si on passe à pydantic + interpolation manuelle, plus clair
  mais break parity.
- **Logging plus structuré** : actuellement log.info des metrics. On
  pourrait émettre des records JSON pour parsing aval.
- **Type hints** : `tuple[StimulusCondition | str, ...]` pour
  `conditions=`, ajouter Literal pour les noms valides.

### Tests à écrire (avant rewrite)

- **Parity bit-exact `pre_train` vs reference** : seed fixé, mêmes
  configs, mêmes losses_1 + losses_2 à 1e-6 près. Tier 1.
- **Parity `evaluate` vs reference `testing()`** : mêmes metrics
  per condition. Tier 1.
- **Test 6-cell vs legacy 2x2 schema** : `BlindsightSetting.from_dict`
  avec les 2 schemas produit le bon BlindsightSetting.
- **Test gradient flow** : vérifier que `loss_2.backward` accumule
  bien dans `fc1.weight.grad` (sinon le `optim_1.zero_grad` order est
  pas important — ce qui contredit notre compréhension).
- **Test asymmetric cascade** : Setting 4 != Setting 6 numériquement
  sur N seeds (sinon le fix D.31 n'a rien changé).
- **Throwaway forward RNG preservation** : runs avec/sans 2nd-order
  ont les mêmes patterns à seed fixé.

### Opportunités perf identifiées

- **Skipper la cascade 1st-order quand `cascade_1st=False`** : déjà
  fait via `cascade_iters_1=1`. ✅
- **Skipper la cascade 2nd-order quand `cascade_2nd=False`** : idem. ✅
- **eval() mode dropout off** : déjà fait via `.eval()`. ✅
- **Vectoriser la boucle d'epochs** : impossible (séquentiel par
  design).
- **Batch generation off-GPU** : actuellement le batch est créé sur
  CPU puis transféré. Si le device est CUDA, on pourrait générer
  directement sur GPU via `torch.rand` au lieu de `np.random`. Casse
  parity bit-exact (RNG différent) mais ~10% speedup.
- **Préallouer les arrays losses_1/2** : déjà fait via `np.zeros(n)`. ✅

### Connexion

- Utilise : `core.cascade.cascade_update` (via networks)
- Utilise : `core.second_order.SecondOrderNetwork`
- Utilise : `core.losses.cae_loss, wagering_bce_loss`
- Utilise : `networks.first_order_mlp.FirstOrderMLP`
- Utilise : `utils.config.load_config`
- Utilise : `domains.blindsight.data.{ConditionParams, StimulusCondition,
  generate_patterns, TrainingBatch}`
- Appelé par : `domains/blindsight/cli.py`

## Méta — pour Claude chat

```text
J'étudie l'architecture de training de la tâche Blindsight du paper
MAPS (Vargas et al. 2025). Une particularité m'interpelle : l'ordre
des optimizer.zero_grad / loss.backward / optimizer.step est non-
standard.

Le pattern habituel pour 2 réseaux séparés serait :

    optim_1.zero_grad()
    loss_1.backward()
    optim_1.step()

    optim_2.zero_grad()
    loss_2.backward()
    optim_2.step()

Mais le paper MAPS fait :

    optim_1.zero_grad()          # ← clear 1st-order grad EARLY
    [run 2nd-order forward]
    optim_2.zero_grad()
    loss_2.backward(retain_graph=True)  # ← 2nd-order loss FLOWS into 1st-order via ComparatorMatrix
    optim_2.step()
    [run 1st-order CAE loss]
    loss_1.backward(retain_graph=True)  # ← adds more grad to 1st-order
    optim_1.step()                       # ← consumes loss_2.grad + loss_1.grad

C'est-à-dire : le 1st-order voit la **somme** des gradients de loss_2
et loss_1, pas seulement loss_1.

Questions :
1. Pourquoi ce design ? C'est paper-faithful (Vargas) mais quel
   bénéfice théorique ?
2. Est-ce équivalent à pondérer la loss totale `α·loss_1 + β·loss_2`
   avec un single optim ?
3. Y a-t-il un papier de référence où ce pattern apparaît (Pasquali
   2010 ?) qui justifie ce co-training ?
```
