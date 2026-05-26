# Walkthrough — Blindsight, du paper au code

**Public visé :** quelqu'un qui a lu `cascade-from-paper-to-code.md`
(Sprint 11) et veut comprendre comment le domaine Blindsight branche
le `core/` MAPS dans un entraînement reproductible.

**Pré-requis :** Sprint 11 closed (commits `c498bb3` à `148fe66`).
Sprint 12 phases A→H (commits `fd4f190` à closeout).

---

## 1. La tâche en deux phrases

**Blindsight** simule le phénomène de Weiskrantz (1986) : un patient
avec une lésion V1 peut **deviner** si un stimulus est présent sans
en avoir conscience. MAPS modélise ça par un first-order qui apprend
à reconstruire le pattern, et un second-order ("wager") qui parie sur
la confiance — l'effet "blindsight" apparaît quand la discrimination
réussit mais le wager reste bas.

## 2. Le piège méthodologique : `multiplier/2`

Le code student génère les patterns ainsi :
- Première moitié du batch : noise-only (pas de stimulus)
- Deuxième moitié : noise + stimulus à un index aléatoire, amplitude
  `U(0, 1) · multiplier`

Le wager target (`order_2_target`) suit la règle :

```python
if pattern[stim_idx] >= multiplier / 2:
    target = [1, 0]  # high wager — stimulus "détecté"
else:
    target = [0, 1]  # low wager — stimulus "subthreshold"
```

**Pourquoi `multiplier/2` ?** Le paper §3.2 ne le dit pas
explicitement. Hypothèse : c'est la médiane de `U(0, multiplier)`,
donc ~50% des stimulus présents sont au-dessus → 50% high wager
balancés. Ce seuil **artificiel** est ce qui crée la dissociation
conscient/inconscient : un stimulus PRÉSENT mais sous le seuil
produit une target "low wager" tout en restant détectable par le
first-order. C'est l'effet blindsight, encodé dans la *donnée*.

`★ Insight ─────────────────────────────────────`
Avant Sprint 12, je pensais que l'effet blindsight était produit
*par l'architecture* (la cascade, le 2nd-order). En réalité, il est
**inscrit dans la cible** via ce seuil multiplier/2. L'architecture
apprend à exploiter une dissociation que la data crée. C'est important
pour interpréter scientifiquement : MAPS ne *découvre* pas le
blindsight, il l'*apprend* à partir d'une supervision déjà dissociée.
`─────────────────────────────────────────────────`

## 3. Le two-loss gradient pattern (load-bearing)

Le cœur du training Blindsight est cette séquence de 6 étapes par
epoch (port direct du student `pre_train`) :

```python
# Étape 1 : Cascade 1st-order (50 itérations à α=0.02 si Setting 6)
h1, h2 = None, None
for _ in range(iters_1):
    h1, h2 = first_order(patterns, h1, h2, rate_1)

# Étape 2 : RAZ des gradients 1st AVANT toute backward
optimizer_1.zero_grad()

# Étape 3 : Cascade 2nd-order (50 itérations à α=0.02 si Setting 6)
wager, comparison = None, None
for _ in range(iters_2):
    wager, comparison = second_order(patterns, h2, comparison, rate_2)

# Étape 4 : Loss 2 backward AVEC retain_graph → flow into 1st-order aussi
loss_2 = wagering_bce_loss(wager, target_wager)
optimizer_2.zero_grad()
loss_2.backward(retain_graph=True)  # accumule gradients dans 1st-order
optimizer_2.step()
scheduler_2.step()

# Étape 5 : Loss 1 backward → s'AJOUTE aux gradients déjà accumulés
loss_1 = cae_loss(W=fc1.weight, x=stim_present, recons_x=h2, h=h1)
loss_1.backward(retain_graph=True)

# Étape 6 : optimizer_1.step utilise la SOMME des deux backwards
optimizer_1.step()
scheduler_1.step()
```

**Pourquoi c'est load-bearing :** l'ordre `optimizer_1.zero_grad()
**avant** loss_2.backward` est crucial. Si on inverse (zero après
loss_2), on perd le gradient cross-task du 2nd-order vers le 1st-order.
Le 1st-order n'apprend alors *que* depuis sa propre loss CAE — il
ne reçoit jamais le signal du wager.

Cette interaction cross-task est probablement *le mécanisme principal
par lequel MAPS apprend à dissocier discrimination et wager*. Si on
training sans cet ordre :
- 1st-order optimise uniquement la reconstruction
- 2nd-order optimise uniquement le wager
- Pas de coupling explicite — chaque réseau ignore l'autre

Avec l'ordre paper-faithful :
- 1st-order voit la SOMME : gradient de reconstruction + gradient
  "ce que le 2nd-order veut que tu encodes"
- Le 1st-order modifie ses features pour aider AUSSI le 2nd-order
- Coupling implicite

`★ Insight ─────────────────────────────────────`
Ce pattern est *non documenté en clair dans le paper*. Il faut
lire le code student pour le découvrir. C'est exactement le genre
de détail méthodologique qu'un reverse-prompt + un Sprint 12.G Tier
4-light parity test *capturent et préservent*. Le test parity Tier
4-light est le filet qui empêche un futur refactor de casser cette
interaction silencieusement.
`─────────────────────────────────────────────────`

## 4. Cascade asymétrique par setting

Sprint 12 implémente le 6-cell schema (D12.6) : `(cascade_1st,
cascade_2nd, second_order)` indépendants. Avant Sprint-08 D.31, le
trainer student appliquait la cascade *symétriquement* — Setting 4
(cascade 1st + 2nd-order) produisait en réalité les chiffres de
Setting 6 (cascade les deux + 2nd-order). Bug fixed D.31.

Notre port fait le dispatch dans `_cascade_params()` :

| Setting | cascade_1st | cascade_2nd | 2nd-order | rate_1 / iters_1 | rate_2 / iters_2 |
|---------|:-----------:|:-----------:|:---------:|:----------------:|:----------------:|
| 1 baseline | ❌ | ❌ | ❌ | 1.0 / 1 | 1.0 / 1 |
| 2 cascade 1st | ✅ | ❌ | ❌ | **0.02 / 50** | 1.0 / 1 |
| 3 2nd-order only | ❌ | ❌ | ✅ | 1.0 / 1 | 1.0 / 1 |
| 4 MAPS (1st) | ✅ | ❌ | ✅ | **0.02 / 50** | 1.0 / 1 |
| 5 cascade 2nd | ❌ | ✅ | ✅ | 1.0 / 1 | **0.02 / 50** |
| 6 Full MAPS | ✅ | ✅ | ✅ | **0.02 / 50** | **0.02 / 50** |

Setting 2 a un effet *nul* sur le 1st-order training (cascade no-op
sans dropout — D-sarl-cascade-noop). Setting 4 et 6 utilisent la cascade
sur le 1st-order *avec* dropout encoder p=0.1 — donc cascade non-triviale
via MC-dropout (cf. walkthrough `cascade-from-paper-to-code.md` §5).

## 5. SimCLR augmentation — D12.4 baseline

Sprint 11.D a porté la math `simclr_loss` (NT-Xent, Chen 2020) mais
sans pipeline d'augmentation. Sprint 12.D fournit le pipeline :

```python
# domains/blindsight/augmentations.py
def bit_flip(x, p=0.1, *, generator=None):
    """Replace p% of elements with fresh uniform [0, 1] noise."""
    if p == 0.0:
        return x.clone()
    mask = torch.rand(x.shape, generator=generator, device=x.device) < p
    noise = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return torch.where(mask, noise, x)
```

Pour les stimulus Blindsight (continus, pas binaires), "bit-flip" est
interprété comme **noise replacement** : on remplace 10% des éléments
par une nouvelle valeur uniforme. C'est le baseline le plus simple
choisi en D12.4.

Le trainer dispatche sur `first_order_loss.kind` :
```python
if loss_kind == "cae":
    loss_1 = cae_loss(W, stim_present, h2, h1, lam)
else:  # simclr
    augmented = bit_flip(patterns, p=0.1)
    _, h2_aug = first_order(augmented, None, None, rate_1)
    loss_1 = simclr_loss(h2, h2_aug, temperature=0.5)
```

**Comparison empirique CAE vs SimCLR** : reportée post-Sprint 12.
Si SimCLR sous-performe drastiquement, revisiter D12.4 options B/C/D.

## 6. La pile à 4 tiers de parity

Sprint 12.G écrit 15 tests parity contre `external/paper_reference/blindsight_tmlr.py` :

| Tier | Quoi | Tolérance | Bug si fail |
|------|------|-----------|-------------|
| **1** | `generate_patterns` bit-exact | 1e-6 | Drift RNG (np.random vs random) |
| **2** | `FirstOrderMLP` forward + backward | 1e-7 (forward), 1e-7 (grad) | Maths encoder/decoder différent |
| **3** | 1 update step (forward+backward+optim.step) | 1e-6 | Gradient flow / detach W |
| **4-light** | `train(n_epochs=2)` losses | 1e-4 | Two-loss order, dropout RNG, scheduler |

Le Tier 4-light est **le filet de sécurité ultime** : si toute la
pile passe, on a confiance que le port reproduit le paper Table 5a
au niveau loss-séquence. Avec le seuil 1e-4, on tolère le bruit
float32 accumulé sur 2 epochs de cascade + dropout + 2 optimizers.

Note importante : Tier 4-light désactive le hidden Pasquali
(`second_order.hidden_dim=0`) pour comparer apples-to-apples avec
le student qui a ce bug D.25. Production runs MAPS-faithful utilisent
`hidden_dim=100` (D.25 restored).

## 7. Ce qui reste à faire (post-Sprint 12)

- **Reproduction empirique** : `make test-slow` ou run manuel
  `make test-slow` Setting 6 Full MAPS × 4-6 seeds → comparer aux
  z-scores paper Table 5a (discrim ~0.97, wager ~0.85).
- **Si gap** : logger dans `docs/reproduction/deviations.md` avec
  hypothèses (RG-002 H6+ à investiguer).
- **Comparaison CAE vs SimCLR** : reproduire Setting 6 avec
  `first_order_loss.kind=simclr` et comparer. Le test smoke Sprint 12.E
  montre que le dispatch fonctionne ; reste à valider que SimCLR
  converge vers des chiffres décents sur Blindsight.

## 8. Connexions

- **Module Blindsight** : `src/maps/domains/blindsight/{data, augmentations, trainer, cli}.py`
- **Networks** : `src/maps/networks/first_order_mlp.py` (réutilisé AGL Sprint 13)
- **Core** : `src/maps/core/{cascade, second_order, losses}.py` (Sprint 11)
- **Utils** : `src/maps/utils/{config, seeding, logging_setup, device, paths}.py`
- **Configs** : `config/{maps.yaml, domains/blindsight/{env, training}.yaml}`
- **Reference** : `external/paper_reference/blindsight_tmlr.py`
- **Tests parity** : `tests/parity/blindsight/` (4 tiers, 15 tests)
- **Reverse-prompts source** : `docs/learning/reverse-prompts/domains/blindsight/`

## 9. Pour aller plus loin

- **`docs/learning/walkthroughs/cascade-from-paper-to-code.md`** — la cascade McClelland (Sprint 11)
- **`docs/learning/glossaire.md`** — vocabulaire MAPS
- **`docs/reproduction/deviations.md`** — D.25, D.31, RG-002
- **Pasquali & Cleeremans (2010)** — origine du second-order + wager
- **Weiskrantz, L. (1986)** — *Blindsight: a case study and implications*
