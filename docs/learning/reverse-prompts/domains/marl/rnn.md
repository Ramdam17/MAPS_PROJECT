# Reverse-prompt — `src/maps/domains/marl/rnn.py`

**Module path actuel :** `src/maps/experiments/marl/rnn.py`
**Taille :** ~150 lignes.
**Paper :** §B.4 Fig.4 RNN block + eq.6 cascade.

---

## 1. Quelle aurait été la spec ?

> Écris les RNN layers MARL : `RNNLayer` (baseline) et `RNNLayerMeta`
> (meta variant). Port student `utils/rnn.py` et `utils/rnn_meta.py`.
>
> **Both classes** :
> - GRU (1 layer, hidden=100)
> - LayerNorm post-GRU
> - Cascade update sur output (paper eq.6)
> - Init : weights orthogonal/xavier, biases zero
>
> **Shared `_rnn_forward(rnn, recurrent_n, x, hxs, masks)`** :
> 2 call modes :
> 1. **Rollout** : `x.size(0) == hxs.size(0)` → single-step forward.
>    Apply mask: `hxs * masks.repeat(1, recurrent_n).unsqueeze(-1)`.
> 2. **Minibatch** : `x` is `(episode_len * batch_num, input_size)`.
>    Split par les transitions où mask=0 (terminal), forward chunks
>    contigus, concat.
>
> **`RNNLayer`** (baseline) :
> - Forward `(x, hxs, masks, prev_cascade, cascade_rate) →
>   (normed_x, hxs, output_cascade)`
> - Output_cascade = cascade_update(x_raw, prev_cascade, cascade_rate)
> - Normed = LayerNorm(x_raw) — note : norm on raw, not cascade output
>
> **`RNNLayerMeta`** (meta variant) :
> - Same structure as RNNLayer
> - **Drop dead `wager=True` branch** (E.4 audit : 0 call sites)
> - Output `normed = LayerNorm(output_cascade)` — note: norm on cascade,
>   different from baseline (subtle but matters for parity)

## 2. Contraintes scientifiques

### Why GRU not LSTM ?

Paper §B.4 silent. Student utilise GRU. Plus simple, moins de params.
LSTM serait viable mais pas porté.

### `recurrent_n = 1` (paper Fig.4)

Single GRU layer. Standard for MAPPO.

### Cascade sur output

Cascade applied to GRU output post-step. `prev_cascade` threaded from
previous call. Active iff caller passes non-None.

### Diff RNNLayer vs RNNLayerMeta

**Subtle but parity-critical** :
- `RNNLayer` : `normed = norm(x_raw)`, `output_cascade = cascade(x_raw,
  prev, α)`. Norm appliquée sur output GRU brut.
- `RNNLayerMeta` : `output_cascade = cascade(x_raw, prev, α)`, `normed
  = norm(output_cascade)`. Norm appliquée sur cascade output.

Différent ! Pourquoi student a fait ça ? Probable bug ou choix
non-intentionnel. Préservé pour parity.

## 3. Contraintes d'ingénierie

### `_rnn_forward` shared helper

Pour éviter de dupliquer la mask-handling logic (qui est la partie
non-triviale) entre les 2 RNN classes.

### Minibatch mask transitions

Pendant PPO update, les batches mix multiple episodes. Mask=0 marks
episode boundaries. Le GRU doit reset son state là.

```python
has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero()
# Split forward in contiguous chunks separated by has_zeros
```

Logique load-bearing pour parity. Reproduit student L60-83.

### Dead `wager=True` branch dropped

Student `RNNLayer_Meta` avait un internal `wager=True` branch (L97-107)
qui n'est jamais appelé (E.4 grep confirmed). Notre port l'omet pour
clean code.

## 4. Deviations / dettes présentes

### Pas de déviation paper

Architecture matches paper Fig.4 (GRU + LayerNorm).

### ⚠️ Subtle norm position diff (load-bearing)

`RNNLayer` norm sur x_raw, `RNNLayerMeta` norm sur cascade_output.
Different. Pas un bug nécessairement, mais à documenter explicitement
car non-intuitif. Voir §2 above.

### Pas de DETTE active

## 5. Questions ouvertes

- **Q1 :** Le diff norm position RNNLayer vs RNNLayerMeta — student
  bug ou intentional ? Si bug, à corriger post-Phase F.

- **Q2 :** Cascade sur GRU output sur 50 iters avec dropout absent →
  no-op ? Need to check. `prev_cascade` est threaded mais si chaque
  iter produit même output GRU, cascade no-op.

- **Q3 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder
- 2 classes RNN + RNNMeta
- Shared `_rnn_forward` helper
- Mask transitions handling
- Drop dead wager branch
- Subtle norm diff (parity)

### À améliorer
- **Documenter explicitly** le norm position diff
- **Test bit-exact** vs reference pour les 2 classes
- **Type hints `nn.GRU`** explicit

### Tests à écrire
- Rollout mode forward shape correct
- Minibatch mode mask transitions
- Cascade thread through forward calls
- Init orthogonal correct

### Connexion
- Utilisé par : `policy.{MAPPOActor, MAPPOCritic, MAPSActor, MAPSCritic}`
- Utilise : `core.cascade.cascade_update` (via `nn.GRU` + custom)
