# Reverse-prompt — `src/maps/domains/sarl/model_v1.py` (Sprint 09)

**Module path actuel (sur main) :** `src/maps/experiments/sarl/model_v1.py`
**Taille :** ~210 lignes.
**Status :** 🟢 **v1 paper-canonical** (Sprint-09 D-sarl-wrong-variant
resolution).
**Paper :** §2.1, §3, Table 6.

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `model_v1.py` qui porte les networks SARL
> **paper-canonical** depuis `external/paper_reference/sarl/maps_v1.py`
> (Juan's fork, pre-commit `8aa1138`).
>
> **Critical context Sprint-09** :
> - `SARL_Training_Standard.sh` (le seul shell qui a produit Table 6
>   paper) invoque `maps_v1.py` avec `base=2_000_000` frames et `-ema 25`
>   (α=0.25)
> - Le port Sprint-04b a anchoré sur `maps_v2.py` (= `model.py`), basé
>   sur TD-008 audit (token-presence diff, pas structural)
> - **v1 et v2 diffèrent sur 4 pièces architecturales structurelles** :
>   - Q-head input dim : 1024 (v1) vs 128 (v2)
>   - Reconstruction decoder : dedicated `Linear(128, 1024)` (v1) vs
>     tied-weight `F.linear(W.t())` (v2)
>   - `comparison_layer` : active 1024×1024 (v1) vs commented-out (v2)
>   - Cascade target : 1024-dim Output (v1) vs 128-dim Hidden (v2)
> - Training budget : 2M frames + α=0.25 (v1, paper) vs 500k + α=0.45 (v2)
>
> **Donc le port v1 est obligatoire** pour reproduire paper Table 6.
> Sprint-09 Phase 9.1 a porté ce module ; tests parity bit-exact OK
> dans `tests/parity/sarl/test_sarl_v1_parity.py`.
>
> **2 classes** :
>
> **1. `SarlQNetworkV1`** (first-order Q-network) :
> - `Conv2d(in_channels, 16, 3, 1)` → ReLU
> - `Linear(1024, 128)` (fc_hidden) → ReLU
> - **`Linear(128, 1024)` (fc_output)** → ReLU — **dedicated decoder**
>   (vs v2 tied-weight). Le `bias=True` default sert de `b_recon` paper
>   eq.12, **pas besoin de param séparé**.
> - **Cascade sur Output (1024-d, post-decoder)**, pas sur Hidden
> - **Q-head reads from Output** : `Linear(1024, num_actions)` (vs v2
>   `Linear(128, num_actions)`)
> - Forward returns `(q_values, hidden, comparison, output)` où le 4e
>   slot est Output post-cascade (1024-d)
>
> **2. `SarlSecondOrderNetworkV1`** :
> - **`comparison_layer = Linear(1024, 1024)` active** (~1.05M params)
>   uniform init `(-1.0, 1.0)`
> - `Dropout(p=0.1)` → cascade_update → `Linear(1024, 2)` wager
> - Forward : `dropout(ReLU(comparison_layer(comparison_matrix)))` →
>   cascade → wager (raw logits)
> - vs v2 : `dropout(comparison_matrix)` direct (pas de linear layer)

## 2. Contraintes scientifiques

### Pourquoi v1 et pas v2 (D-sarl-wrong-variant)

`SARL_Training_Standard.sh` :
```bash
python maps_v1.py --game $game --seed $seed --steps $base --ema 25 ...
```

`base=2000000` frames, `-ema 25` (α=0.25). Ce sont les paramètres
EXACTS qui ont produit Table 6 paper.

Le port Sprint-04b a anchoré sur `maps_v2.py` (sans s'en rendre compte
car TD-008 audit était une token-presence diff, pas structural). v2
est utilisé par Juan mais pas pour Table 6.

Conclusion : **v1 canonical, v2 wrong variant**.

### Architecture v1 — paper §2.1 eq.1+12

- eq.1 : `C = X - Ŷ` (comparator residual)
- eq.12 : `Ŷ = ReLU(W_dec · Hidden + b_recon)`

v1 implémente eq.12 avec `fc_output = nn.Linear(128, 1024)` (bias
inclus). v2 utilise tied-weight + b_recon séparé.

Architecturalement, v1 = **standard MLP** ; v2 = **tied autoencoder**.

### Cascade sur Output 1024-d

v1 cascade sur le **decoder output** (post-reconstruction), pas sur
hidden. Conceptuellement, c'est la cascade des **activations finales**
(la reconstruction), pas du hidden bottleneck.

**Différence sémantique majeure** avec v2.

### `comparison_layer` active

v1 a `nn.Linear(1024, 1024)` dans le 2nd-order. Ça donne ~1.05M params
de plus que v2. C'est ce qui rend le 2nd-order v1 vraiment expressif
(2nd-order v2 est trivial : juste dropout + wager Linear).

Init uniform `(-1.0, 1.0)` matches student. Large init range
(comparable au FirstOrderMLP).

## 3. Contraintes d'ingénierie

### Layer construction order

```python
self.conv = nn.Conv2d(...)
self.fc_hidden = nn.Linear(1024, 128)
self.fc_output = nn.Linear(128, 1024)
self.actions = nn.Linear(1024, num_actions)
```

Ordre matters pour init RNG. Différent de v2 (`actions = Linear(128,
num_actions)`).

### Tests parity tier 1/3

`tests/parity/sarl/test_sarl_v1_parity.py` asserte bit-exact via
`torch.allclose(ref_out, ours, atol=1e-6)` après `load_state_dict`.

### Import dependencies

`NUM_LINEAR_UNITS` partagé via import de `model.py` (v2 module). Pas
de duplication.

## 4. Deviations / dettes présentes

### 🚨 D-sarl-wrong-variant (Sprint 09 resolved)

C'est CE module qui résout la déviation. **Sprint 09 Phase 9.1 = port
v1**. Phase 9.2 = wiring dans training_loop. Phase 9.5 = MARL align +
deviations.md update.

**Phase F doit être re-run sur Narval avec v1 à 2M frames** post-merge
PR Sprint 09 — actuel TODO post-Sprint 10.

### ✅ DETTE-1 (cross-ref)

v1 a `SarlSecondOrderNetworkV1` avec comparison_layer active. **Plus
proche du `core/SecondOrderNetwork`** que v2 (qui n'a pas de
comparator). Unification post-Phase F serait plus naturelle.

### D-sarl-cascade-noop (v1 same as v2)

Cascade sur Output v1 — pas de dropout dans la chain conv→hidden→output
→ même no-op math. Documented in deviations.md.

## 5. Questions ouvertes

- **Q1 :** Pourquoi v2 a été créé par Juan ? Probable hypothesis :
  tied-weight = moins de params, idée d'efficient autoencoder. Mais
  n'a pas été utilisé pour Table 6. À vérifier si v2 est cité quelque
  part dans le paper (annexes ?).

- **Q2 :** Avec `comparison_layer` active (v1), le 2nd-order a ~1.05M
  params. C'est beaucoup. Est-ce que le wager apprend vraiment des
  features pertinents ou est-ce de l'overfitting ?

- **Q3 :** Cascade sur Output 1024-d v1 — comment ça interagit avec
  le no-op finding ? Le cascade averaging était utile sur path
  dropout (Blindsight 2nd-order). Ici Output est déterministe → no-op
  pareil que v2. Donc Phase F pourrait montrer Setting 2 ≈ Setting 1
  pour v1 aussi.

- **Q4 :** *(à toi)*

## 6. Pour le rewrite (Sprint 11+)

### À garder (paper-canonical)
- Dedicated `fc_output` decoder
- Cascade sur Output
- Q-head from Output
- `comparison_layer` active dans 2nd-order
- Forward signature returns 4-tuple

### À améliorer
- **Docstring explicit** : mention "v1 canonical, v2 secondary, see
  D-sarl-wrong-variant"
- **Asserts shape** dans forward pour catch refactor drift
- **Naming `prev_h2`** ambigu (en v1 c'est Output, en v2 Hidden) —
  peut-être renommer `prev_cascade_state` pour clarté

### Tests à écrire
- Parity bit-exact vs `maps_v1.py` reference (déjà existant Sprint
  09)
- Forward shapes : q (B, A), hidden (B, 128), comparison (B, 1024),
  output (B, 1024)
- comparison_layer init range correct (-1.0, 1.0)
- fc_output bias = 0 par défaut (PyTorch default), apprend pendant
  training

### Opportunités perf
- v1 cascade sur Output 1024-d → 50 iters × Linear(128→1024) →
  ~6.5M ops par iter. **Coûteux**. Mais avec no-op finding, un seul
  forward suffirait.
- Si on adoptait "skip cascade quand path déterministe" (`core/cascade.md`
  Opt-C1-A), v1 Setting 2/4 deviendrait ~50× plus rapide localement.

### Connexion
- Importe `NUM_LINEAR_UNITS` from `model.py` (v2)
- Utilisé par `training_loop.py` quand `model_variant="v1"` (default
  Sprint 09+)
- Tests parity tier 1/3 dédiés v1

## Méta — pour Claude chat

```text
J'analyse une déviation critique dans la reproduction d'un paper de
recherche : on a porté la mauvaise variante d'un network DQN.

Variant v1 (paper canonical, produced Table 6 numbers) :
- Dedicated decoder Linear(128, 1024) — standard autoencoder
- Cascade sur output (1024-d post-decoder)
- Q-head reads from output (1024-d)
- Active comparison_layer 1024×1024 dans 2nd-order
- Training budget 2M frames, α EMA=0.25

Variant v2 (Sprint-04b port, used for v2 archived runs) :
- Tied-weight decoder F.linear(hidden, fc_hidden.weight.T) + b_recon
- Cascade sur hidden (128-d compressed)
- Q-head reads from hidden (128-d)
- No comparison_layer dans 2nd-order (juste dropout)
- Training budget 500k frames, α EMA=0.45

Questions :
1. Quel est le rationale architectural pour porter de tied-weight ?
   Bénéfice = moins de params. Coût = couples encoder/decoder weights
   (le Jacobian Rifai n'a plus la même sémantique).
2. Cascade sur hidden vs output — sémantiquement très différent. Le
   hidden est un bottleneck compressé (128-d), output est la
   reconstruction (1024-d). Pourquoi choisir l'un vs l'autre ?
3. Avec 2 architectures structurellement distinctes, peut-on comparer
   honnêtement les résultats numériques ? Ou faut-il les voir comme
   2 papiers différents ?
```
