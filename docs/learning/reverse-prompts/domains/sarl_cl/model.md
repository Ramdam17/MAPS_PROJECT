# Reverse-prompt — `src/maps/domains/sarl_cl/model.py`

**Module path actuel (sur main) :** `src/maps/experiments/sarl_cl/model.py`
**Taille :** ~260 lignes.
**Paper :** §4 (continual learning).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `model.py` pour SARL+CL. **3 networks
> distincts** du standard SARL — duplication intentionnelle car
> structuralement différents. Port `SARL_CL/examples_cl/maps.py:117-219`.
>
> **1. `SarlCLQNetwork`** : similar to **v1 SARL** mais explicit :
> - `Conv(in_channels, 16, 3, 1)` → ReLU
> - `Linear(1024, 128)` (fc_hidden) → ReLU
> - **`Linear(128, 1024)` (fc_output) dedicated decoder** → ReLU
>   (PAS tied-weight)
> - **Cascade sur Output (1024-d)**, pas sur Hidden
> - **Q-head from Output** : `Linear(1024, num_actions)`
> - Includes `self.sigmoid = nn.Sigmoid()` attribute as paper-code
>   artifact (never called in forward, kept pour state_dict parity
>   legacy checkpoints)
> - Forward `(x, prev_h2, cascade_rate) → (q_values, hidden,
>   comparison, output)`. `comparison = flat_input - output`.
>
> **2. `SarlCLSecondOrderNetwork`** : differs de standard SARL :
> - **`comparison_layer = Linear(1024, 1024)` active** uniform init
>   `[-1, 1]`
> - Forward : `dropout(ReLU(comparison_layer(comparison_matrix)))` →
>   cascade → wager
> - Wager `Linear(1024, 2)` init `[0, 0.1]`
> - Returns raw logits (B, 2)
>
> **3. `AdaptiveQNetwork`** : key innovation for **cross-game
> transfer** (curriculum learning multi-game) :
> - Topology identical à `SarlCLQNetwork`
> - **Prepended 1×1 conv adapter** : `nn.Conv2d(max_input_channels, 16,
>   1)` puis le pipeline standard
> - **Zero-padding pre-step** : si l'input a `< max_input_channels`,
>   zero-pad sur la dim channels avant l'adapter
> - Permet de charger un teacher entraîné sur Breakout (4 channels) et
>   continuer sur Seaquest (10 channels) sans rebuilding from scratch
> - `max_input_channels=10` paper Table 11 CL (D-sarl_cl-max-channels
>   D.20 resolved)
>
> **Ne PAS refactor** pour partager layers avec `sarl.model`. Les archis
> diffèrent vraiment ; silent unification changerait les Tables paper
> CL.

## 2. Contraintes scientifiques

### Cross-game transfer challenge

MinAtar games ont des **channel counts différents** :
- Breakout : 4 channels
- Space Invaders : 6 channels
- Asterix : 4 channels
- Freeway : 7 channels
- Seaquest : 10 channels (le plus large)

Un Q-network trained sur Breakout (`Conv(4, 16)`) ne peut pas être
réutilisé directement sur Seaquest (`Conv(10, 16)`). Il faudrait
rebuild le première conv layer → perdre tout l'apprentissage.

**Solution AdaptiveQNetwork** : utiliser `max_input_channels=10` par
défaut, et zero-pad les inputs <10. Le 1×1 adapter conv apprend à
projeter les channels présents.

### Paper §4 continual learning

Le paper §4 montre que sans CL machinery, un agent qui apprend
sequentially Game1 → Game2 → Game3 → Game4 **oublie** Game1 (catastrophic
forgetting, McCloskey & Cohen 1989).

MAPS ajoute :
1. **Adaptive backbone** (channel padding) → permet transfer
2. **Teacher distillation** (L2 weight anchor + feature MSE) → freine
   forgetting
3. **Curriculum chaining** (4 games × 100k frames chacun = 400k total)

## 3. Contraintes d'ingénierie

### `_in_channels` stored mais unused

`SarlCLSecondOrderNetwork._in_channels` est stocké mais jamais utilisé
dans forward. Kept for API parity with paper constructor signature.

### `sigmoid` attribute jamais appelé

`SarlCLQNetwork.sigmoid = nn.Sigmoid()` défini dans `__init__` mais
non appelé dans `forward`. Paper-code artifact, kept pour
checkpoint loading parity (state_dict keys cohérent).

### Layer construction order

Conv → sigmoid → fc_hidden → fc_output → actions. Différent du SARL
v1/v2. Affecte init RNG.

### `AdaptiveQNetwork` channel-pad logic

```python
if x.shape[1] < self.max_input_channels:
    pad_size = self.max_input_channels - x.shape[1]
    x = F.pad(x, (0, 0, 0, 0, 0, pad_size))  # pad channels dim
x = self.adapter_conv(x)
# ... rest pipeline
```

Conserve les semantics : input réel projeté + zeros pour les channels
manquants.

## 4. Deviations / dettes présentes

### D-sarl_cl-max-channels (D.20 resolved)

Paper Table 11 CL : `max_input_channels=10`. Student port avait 7.
Sprint-08 D.20 aligné à 10. Couvre Seaquest.

### D-sarl_cl-channel-adapter (D.17 to verify)

Paper p.9 dit "1×1 conv + ReLU" comme adapter. Student utilise
"zero-padding + max". À vérifier que l'impl matche le paper claim.

### DETTE-1 cross-ref

`SarlCLSecondOrderNetwork` vs `SarlSecondOrderNetwork` (SARL v2) :
- CL a `comparison_layer` active (comme SARL v1)
- v2 SARL ne l'a pas
- Donc CL est conceptuellement plus proche de v1 que v2

Unification post-Phase F : potentiellement on partage le `SecondOrderCore`
entre v1 et CL.

### Pas de DETTE active sur ce module

## 5. Questions ouvertes

- **Q1 :** L'`AdaptiveQNetwork` est-il efficace pour TOUS les
  transfers ? Padding zero pour les channels manquants peut être
  sous-optimal — le 1×1 adapter ne voit pas ces channels comme
  "manquants" mais comme zeros. Alternative : adapter conv trainable
  par game ?

- **Q2 :** Le `sigmoid` artifact non-appelé est un code smell. Il a
  un `Parameter` dans state_dict (potentially). À supprimer ?
  Casse legacy checkpoint loading.

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- 3 classes distinctes (CL spécifique)
- Dedicated decoder (vs tied) — comme SARL v1
- Cascade sur Output
- `comparison_layer` active in 2nd-order
- AdaptiveQNetwork with channel padding
- `max_input_channels=10` default

### À améliorer
- **Supprimer `sigmoid` artifact** si on confirme aucun legacy
  checkpoint l'utilise
- **Type hint `Optional[int]`** pour `max_input_channels`
- **Documenter explicitement** le channel padding strategy

### Tests à écrire
- Bit-exact parity vs paper reference
- `AdaptiveQNetwork(in_channels=4, max_input_channels=10)` accepte
  inputs (B, 4, 10, 10), pad à (B, 10, 10, 10)
- Forward shapes : q (B, A), hidden (B, 128), comparison (B, 1024),
  output (B, 1024)

### Opportunités perf
- Channel pad via `torch.cat([x, zeros(...)], dim=1)` vs `F.pad` —
  équivalent, pas de gain
- Cascade sur Output 1024-d : ~50× les ops d'un cascade 128-d. Pas
  trivial à optim sans casser parity.

### Connexion
- Utilise : `core.cascade.cascade_update`
- Utilisé par : `domains/sarl_cl/training_loop.py`, `trainer.py`
