# Reverse-prompt — `src/maps/domains/marl/util.py`

**Module path actuel :** `src/maps/experiments/marl/util.py`
**Taille :** ~110 lignes.

---

## 1. Quelle aurait été la spec ?

> Écris les low-level helpers ports de student `onpolicy/algorithms/
> utils/util.py` et `onpolicy/utils/util.py`. Scope E.5 lock : drop
> tout sauf ce qui est used par le MAPS-faithful MARL port.
>
> **Functions** :
>
> - **`init(module, weight_init, bias_init, gain=1.0)`** : apply
>   weight_init + bias_init to module. Used by Categorical / v_out /
>   CNNLayer pour combiner les 2 initializers en helper compact.
>
> - **`check(x)`** : convert np.ndarray → tensor if needed.
>
> - **`calculate_conv_params(input_size)`** : heuristic kernel/stride/
>   padding for ConvLayer. `kernel=5 if >100 else 3, stride=1,
>   padding=(kernel-1)//2`.
>
> - **`get_shape_from_obs_space(obs_space)`** : extract shape from
>   gymnasium Box/Discrete spaces.
>
> - **`huber_loss(diff, delta=10.0)`** : `0.5·diff² if |diff|<=δ else
>   δ·(|diff|-0.5·δ)`. Used par MAPPOTrainer.cal_value_loss.
>
> - **`mse_loss(diff)`** : `0.5·diff²`.
>
> - **`update_linear_schedule(optimizer, epoch, total_epochs, initial_lr)`** :
>   simple linear LR decay. Used pendant PPO outer loop.
>
> - **`get_grad_norm(parameters)`** : compute the L2 grad norm across
>   all params. Used pour logging.
>
> **Dropped** :
> - `weight_init` (covers LSTM/Embedding/MultiHead — not used)
> - `get_clones` (attention extensions — not used)
> - `get_shape_from_act_space` MultiDiscrete/MultiBinary/Box paths
>   (Discrete only)
> - `tile_images` (rendering, not used)

## 2. Constraintes d'ingénierie

### `calculate_conv_params` heuristic

```python
height, width, _ = input_size
kernel_size = 5 if (height > 100 or width > 100) else 3
stride = 1
padding = (kernel_size - 1) // 2
```

Pour MeltingPot 11×11×3 → kernel=3, stride=1, padding=1.

### `huber_loss` with `delta=10.0`

Standard PPO uses δ=1. Student MAPPO uses δ=10. Plus permissif.

Effect : pour |diff| > 10, gradient = ±10 (clamped). Pour |diff| ≤
10, gradient = diff (linear). Plus tolerant aux outliers.

### `update_linear_schedule` standard

```python
lr = initial_lr - (initial_lr * epoch / total_epochs)
```

Linear decay to 0 over training. Standard PPO.

## 3. Deviations / dettes présentes

Pas de déviation paper.

**Pas de DETTE active.**

## 4. Questions ouvertes

- **Q1 :** `huber_loss delta=10` plus permissif que standard 1. Effect
  sur convergence ?

- **Q2 :** `calculate_conv_params` heuristic only 2 modes (3×3 vs 5×5).
  Si on a 256×256 inputs un jour ?

## 5. Pour le rewrite (Sprint 11+)

### À garder
- Tous les helpers documented
- `delta=10` Huber (parity student)
- Linear LR schedule

### À améliorer
- **Type hints** sur tous les helpers
- **Test bit-exact** vs reference

### Tests à écrire
- `huber_loss(0)` = 0
- `huber_loss(11, delta=10)` linear branch
- `calculate_conv_params(11, 11, 3)` returns (3, 1, 1)
- `update_linear_schedule` decay correct

### Connexion
- Utilisé par : `policy, trainer, encoder, act, data` (basically all
  MARL modules)
