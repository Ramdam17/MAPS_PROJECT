# Reverse-prompt — `src/maps/networks/first_order_mlp.py`

**Module path actuel (sur main) :** `src/maps/networks/first_order_mlp.py`
**Taille :** 121 lignes, 1 classe (`FirstOrderMLP`) + 2 helpers
(`_global_sigmoid`, `make_chunked_sigmoid`).
**Paper :** Table 9 (Blindsight), Table 10 (AGL), §2.2 + §2.3.
**Review existant :** `docs/reviews/first_order_mlp.md` (Sprint-08 C.11).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `first_order_mlp.py` qui implémente le backbone
> autoencoder-style partagé entre Blindsight et AGL. Architecture :
>
> - **Encoder** : `Linear(input_dim, hidden_dim, bias=False) → ReLU →
>   Dropout(0.1)`
> - **Decoder** : `Linear(hidden_dim, input_dim, bias=False) →
>   decoder_activation(...)`
> - **No bias anywhere** (matches student exactement)
> - **Weight init uniforme** `(-1.0, 1.0)` sur les 2 couches
>
> Le `decoder_activation` est paramétré parce que les deux domaines
> diffèrent :
> - **Blindsight** : sigmoid global sur les 100 dims (`_global_sigmoid`)
> - **AGL** : sigmoid par chunk de 6 bits (`make_chunked_sigmoid(6)`) —
>   chaque chunk encode une lettre, l'output approxime un one-hot
>
> Le `make_chunked_sigmoid(n)` factory clone le tensor et applique
> sigmoid in-place sur chaque chunk de `n` bits. Student modifie le
> tensor in-place directement — notre version clone d'abord pour éviter
> les pitfalls autograd PyTorch.
>
> Forward : `(x, prev_h1=None, prev_h2=None, cascade_rate=1.0) → (h1,
> h2)`. La cascade s'applique sur **h2 seulement** (decoder output).
> `prev_h1` est accepté pour symétrie API avec le student mais n'est
> **pas utilisé** — Blindsight/AGL ne cascadent pas l'encoder. Garder
> le slot pour permettre future cascade policy.
>
> SARL n'utilise pas ce module (conv-based `SarlQNetwork` à la place).

## 2. Contraintes scientifiques

### Architecture paper

- **Blindsight (Table 9)** : input=output=100, **hidden=60** (paper),
  hidden=100 (config actuel — voir D-blindsight-hidden-40).
- **AGL (Table 10)** : input=output=48, hidden=40.
- **Encoder ReLU + Dropout(0.1)** : student `blindsight_tmlr.py:155, 177`,
  `agl_tmlr.py:152, 171`.
- **Decoder sigmoid** : global pour Blindsight, chunked-per-6-bits pour
  AGL (paper §2.3 — chaque lettre = 6 bits WTA).
- **Weight init uniform(-1, 1)** : student L173-174 Blindsight, L167-168
  AGL.

### Cascade asymétrique (encoder vs decoder)

Le paper §2.1 dit que la cascade s'applique sur la "dernière hidden
layer". Implémentation choisit **h2 (decoder output)** comme cible
cascade, pas h1 (encoder output). Cohérent avec student.

## 3. Contraintes d'ingénierie

### Encoder-decoder tied par CAE loss

La symétrie encoder/decoder force le hidden representation à reconstruire
l'input. C'est exploité par `cae_loss` :
- `W` = `fc1.weight` (encoder)
- `h` = encoder output (post-ReLU, post-dropout)
- `recons_x` = `fc2(h)` puis activation
- Le terme contractif `||J_h||²_F = Σ (h(1-h))² · Σ W²` régularise
  l'encoder via le jacobien.

### Chunked sigmoid : in-place vs clone

Student AGL fait `h2[:, i:i+n] = sigmoid(h2[:, i:i+n])` in-place.
Notre port clone avant pour éviter les autograd issues PyTorch (in-place
sur tensor utilisé upstream peut casser le backward). **Numériquement
identique, plus robuste.**

### Forward `prev_h1` non utilisé

Le param `prev_h1` est dans la signature pour symétrie avec student
(qui aussi le passe sans s'en servir) et pour permettre future cascade
sur l'encoder. C'est un **slot API** vide — pas un bug.

## 4. Deviations / dettes présentes

### 🚨 D-blindsight-hidden-40 (RG-002 H1, résolu D.25)

Paper Table 9 dit `hidden=60`. Student code `blindsight_tmlr.py:2222`
passe `hidden=40` (literal) à `train()`. Config port utilisait
`hidden=100`. Sprint-08 D.25 a tranché : **`hidden=40` aligné avec le
code student** (le code qui a produit les Tables paper). Paper Table 9
est inconsistant avec le code.

Cette résolution + restauration du Pasquali hidden (D-blindsight-wager-
hidden) ont fermé 96% du gap discrim et 86% du gap wager.

### 🚨 Quirk universel `h(1-h)` sur ReLU output (lié à `cae_loss`)

La formule `h(1-h)` est la dérivée de sigmoid. Les 3 domaines (Blindsight,
AGL, SARL) appliquent cette formule sur `h = ReLU(Wx + b)` — output
ReLU. **Mathématiquement faux** mais **universellement présent** dans
le code student → préservé byte-for-byte. Voir `docs/reviews/cascade.md`
et `docs/learning/reverse-prompts/core/losses.md` pour la discussion.

### D-001/D-002 cross-refs

D-001 (wager units) : impact via le 2nd-order, pas ce module.
D-002 (CAE vs SimCLR) : `cae_loss` consomme l'output de ce module
(h1, recons_x). Si on portait SimCLR, on changerait le training mais
pas ce module.

## 5. Questions ouvertes

- **Q1 :** L'absence de bias dans les Linear est-elle paper-faithful
  documentée, ou un choix défaut PyTorch qu'on n'a pas remarqué ? Test :
  ajouter bias → quel z-score ?
- **Q2 :** Le `weight_init_range=(-1, 1)` est large. Glorot/Xavier
  donneraient `(-√(6/(in+out)), √(6/(in+out)))` ≈ `(-0.2, 0.2)`. Pourquoi
  ce choix moins serré ? Convention paper silencieuse.
- **Q3 :** *(à toi, Rémy)* — quel choix ici te semble le plus surprenant ?

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Pas de bias, weight init uniform(-1, 1), encoder ReLU + Dropout(0.1)
- `decoder_activation` paramétré (Blindsight global vs AGL chunked)
- `make_chunked_sigmoid` factory avec clone-then-assign
- Cascade sur h2 uniquement, slot `prev_h1` vide

### À améliorer
- **Type `decoder_activation`** : `Callable[[Tensor], Tensor]` est OK
  mais on pourrait typer plus précisément avec `Protocol` ou un alias.
- **Hidden_dim default** : `40` est AGL-specific. Une callsite Blindsight
  qui oublie de passer `hidden_dim=` aura silencieusement un réseau
  AGL-sized. **Option** : enlever le default, forcer explicit.
- **Documenter explicitement** que `prev_h1` est un slot vide.

### Tests à écrire
- Forward shape : (B, 100) → (h1: B×60 ou B×40, h2: B×100)
- AGL chunked sigmoid sur 48 dims = 8 chunks de 6 bits, chaque chunk
  ∈ [0,1] indépendamment.
- Weight init : valeurs ∈ (-1, 1), distribution approximativement uniforme.
- Cascade sur h2 : threading prev_h2 fait converger.
- Bit-exact parity : output identique à `blindsight_tmlr.py:FirstOrderNetwork`.

### Opportunités perf
- Make_chunked_sigmoid pourrait être vectorisé via `reshape` :
  `h.reshape(B, -1, n).sigmoid().reshape(B, -1)`. Plus rapide qu'une
  boucle Python sur les chunks. Test parity bit-exact requis.

### Connexion
- Utilise `core.cascade.cascade_update`
- Output `h1` consommé par `core.losses.cae_loss` (via trainer)
- Output `h2` consommé comme `recons_x` par `cae_loss`, et comme input
  au `core.second_order.ComparatorMatrix`

## Méta — pour Claude chat

```text
J'étudie une archi autoencoder-style 2-layer MLP utilisée dans le paper
MAPS pour Blindsight (input=output=100) et AGL (input=output=48). Choix
de design :
- No bias (linear layers sans biais)
- Weight init uniforme(-1, 1) — plus large que Glorot/Xavier
- ReLU encoder + Dropout(0.1) puis sigmoid decoder

Le hidden_dim diffère par domaine : Blindsight=60 ou 40 (paper Table 9
dit 60, code dit 40), AGL=40 (Table 10).

Pourquoi ces choix non-standard (no bias, init large) plutôt que les
defaults PyTorch (bias=True, Kaiming init) ? Y a-t-il une justification
théorique dans la lignée Pasquali-Cleeremans / Vargas, ou est-ce
empirique ?
```
