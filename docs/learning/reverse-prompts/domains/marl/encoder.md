# Reverse-prompt — `src/maps/domains/marl/encoder.py`

**Module path actuel :** `src/maps/experiments/marl/encoder.py`
**Taille :** ~134 lignes.
**Paper :** §B.4, Fig.4 ConvEncoder.

---

## 1. Quelle aurait été la spec ?

> Écris `CNNBase` + `CNNLayer` pour encoder les observations RGB
> MeltingPot (11×11×3 après downsample 8×). Port student
> `external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/cnn.py`.
>
> **Pipeline (paper Fig.4)** :
> 1. Normalize RGB : `x / 255.0`
> 2. Permute (H, W, C) → (C, H, W) si needed
> 3. `Conv2d(C, hidden//2, kernel=3, stride=1)` + BatchNorm2d + ReLU
> 4. Flatten + `Linear(flat, hidden) + LayerNorm + ReLU`
> 5. `Linear(hidden, hidden) + LayerNorm + ReLU`
>
> Output `(batch, hidden_size)`. Les 2 extra Linear+LayerNorm post-conv
> viennent du student et ne sont pas individuellement dans paper
> Fig.4, mais le ConvEncoder global y est.
>
> **Dropped** : MLPBase, Encoder, ResidualBlock, Perceiver (unused
> par MAPPO).
>
> **Init** : `init_method = [xavier_uniform, orthogonal][use_orthogonal]`,
> gain = `calculate_gain("relu" if use_ReLU else "tanh")`.
>
> **Channel auto-detect** : si `obs_shape[0] == 3` → CHW, si
> `obs_shape[2] == 3` → HWC, sinon raise.

## 2. Contraintes scientifiques

### Paper Fig.4 Conv → MLP → RNN

Le ConvEncoder dans paper Fig.4 est juste un block "ConvEncoder" →
hidden representation. Les détails (combien de Linear post-conv,
LayerNorm) sont implementation choices.

Notre port garde student exactement : 2 Linear+LayerNorm post-conv.

### Hidden size = 100 (paper Table 12)

Sortie 100-d (post les 2 Linear). Match paper.

### Pas de BatchNorm sur Linear

Student utilise `LayerNorm` (not BatchNorm) sur les Linear layers. Plus
stable en RL où les batches sont small.

## 3. Contraintes d'ingénierie

### Channel auto-detect via heuristic

```python
if obs_shape[0] == 3: CHW
elif obs_shape[2] == 3: HWC
else: raise
```

Marche pour MeltingPot (RGB = 3 channels). Échouerait pour grayscale
ou multi-channel non-RGB. Acceptable scope.

### `calculate_conv_params` utility

Heuristic stride/kernel selection :
- `kernel=5 if (H or W) > 100 else 3`
- `stride=1`
- `padding=(kernel-1)//2`

Pour 11×11 → kernel=3, stride=1, padding=1. Output stays 11×11 (with
padding). Mais student code utilise no padding → output 9×9. To verify.

### init lambda factory

```python
def init_(m):
    return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
```

Pattern : pré-bake init_method + bias_init + gain → re-use sur tous
les modules. Concise.

## 4. Deviations / dettes présentes

Pas de déviation paper sur ce module.

**Pas de DETTE active.**

## 5. Questions ouvertes

- **Q1 :** Paper Fig.4 ConvEncoder = 1 conv block. Notre port = 1
  conv + 2 Linear post. Diff matters ?

- **Q2 :** 16 filters dans conv → MAPS standard. Pas overkill pour
  11×11 input ? Réduire à 8 ?

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Conv + 2 Linear pipeline (paper-faithful via student)
- Auto-detect CHW vs HWC
- LayerNorm (vs BatchNorm)
- Orthogonal init default

### À améliorer
- **Type hints explicits** sur `obs_shape: tuple[int, int, int]`
- **Validate obs_shape == (H, W, 3) or (3, H, W)** en assert
- **Documenter padding=0 vs paper Fig.4** spatial dim drop

### Tests à écrire
- Forward shape : (B, 11, 11, 3) → (B, 100)
- Forward shape : (B, 3, 11, 11) → (B, 100) (CHW)
- Init orthogonal range correct
- BatchNorm on conv output only

### Connexion
- Utilisé par : `policy.MAPPOActor, MAPPOCritic` (et MAPS variants)
- Utilise : `util.{calculate_conv_params, init}`
