# Component review — `src/maps/experiments/sarl/losses.py`

**Review sub-phase :** Sprint-08 D.11 (per-experiment consolidated view ;
cross-component audit already done in C.7/C.10).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl/losses.py` (98 L, 1 function).
**Paper sources :** §3 (SARL), paper Table 11 Jacobian reg weight = 1e-4 (student value, paper silent).
**Student sources :** `external/paper_reference/sarl_maps.py:330-379` (`CAE_loss`).

**Callers du port :**
- `sarl/trainer.py:51, 188` (first-order loss via `loss = cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)`).
- `sarl_cl/trainer.py:64, 283, 302` (CL variant, even under meta + teacher loss composition).

**DoD D.11** : doc créé, cross-ref C.7/C.10/D.3 consolidée, 1 finding docstring mineur flaggé,
0 code touché.

---

## Status post-C.7/C.10 (rappel)

Déjà audité côté cross-component (C.7, C.9 audit callers, C.10 batch fix) :
- **DETTE-2** tracée : doublon `components.cae_loss` (BCE recon pour Blindsight/AGL) vs
  `sarl.losses.cae_loss` (Huber recon pour SARL). Architecturalement justifié par tied-weight
  decoder SARL.
- **D-002 SimCLR vs CAE** : paper eq.4 = SimCLR contrastif (Chen 2020) ≠ student/port = CAE
  (Rifai 2011). Décision queued Phase D.22b (SimCLR impl vs keep CAE).
- **Docstring state_dict claim** corrigée C.10 (port/student équivalents via default `keep_vars=False`).
- **ReLU + h(1-h) quirk** documenté (formula valide sigmoid, appliquée sur ReLU — paper-faithful
  student parity universel, C.11 erratum fix).

## (a) Per-experiment scope — SARL specificities

### Huber reconstruction term

Port L88 : `mse = F.huber_loss(recons_x, x)`.

- **Default `delta=1.0`** (smooth L1 au-delà), **default `reduction="mean"`**. Student `sarl_maps.py:354`
  : `f.huber_loss(recons_x, x)` — mêmes defaults.
- Paper Huber (1964) cité L26 — standard δ=1 non-explicité mais convention historique.
- **Vs Blindsight/AGL** : ces domaines utilisent `components.cae_loss` avec `recon="bce_sum"`
  (BCE sum-reduction sur outputs sigmoid). Asymmetry paper-faithful : SARL `recons_x = Q_s_a`
  (scalar Q-value, not a binarized pattern) → BCE inapplicable.

✅ **Bit-parity student** sur Huber defaults.

### Jacobian regularizer term

Port L92-95 :
```python
dh = h * (1 - h)          # (N_batch, N_hidden) — sigmoid-derivative formula
w_sum = torch.sum(W**2, dim=1).unsqueeze(1)  # (N_hidden, 1)
contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)  # (1,)
```

Student `sarl_maps.py:357-362` identique en structure. ✅ **Bit-parity sur 3 ops.**

### Call signature trompeuse

Port docstring L6-17 documente la **confusion naming** :

```python
loss = cae_loss(W, target, Q_s_a, h1, lam)
#              ↑  ↑       ↑      ↑   ↑
#              W  x       recons h   lam
```

`x = td_target` (bootstrapped reward + γ·Q_target), `recons_x = Q_s_a` (predicted Q). Ce n'est
**pas une reconstruction** au sens Rifai (x = input, recons_x = decoder output). **C'est un DQN
Huber TD error** + regularisation Jacobienne sur `fc_hidden`.

Port docstring L15-17 : *"The 'contractive autoencoder' naming is historical; the loss here is
really `Huber(Q, target) + λ · Jacobian regularization on fc_hidden`."*

✅ **Naming préservé pour parity paper** ; confusion clairement documentée.

## (b) 🚨 Finding D11-F1 — docstring `cae_loss` function-level contradict module-level

Module docstring L32-41 (après C.10 correction) dit :
> **Important:** PyTorch's ``state_dict(keep_vars=False)`` (the default) returns **detached**
> tensors [...] the Jacobian term does **not** propagate gradient directly back to
> ``fc_hidden.weight``

Function docstring L68-69 dit :
> ``W`` : Tensor of shape (N_hidden, N_input). Weight matrix of the hidden layer; passed live from
> ``state_dict()`` **so gradients flow back through it**.

→ **Contradiction directe**. Le function-level docstring est **incorrect** post-C.10 (reste un
résidu oubié). Le correct est : *"passed live from state_dict() which detaches by default;
gradient does NOT flow back through W in the Jacobian term."*

**Piste D11-fix-1** : corriger L68-69 pour matcher module-level. 2 min. Propagé quand je batch
D.11+next fixes, ou en standalone.

## (c) Variable name `mse` pour `huber_loss` — cosmétique

Ligne 88 : `mse = F.huber_loss(recons_x, x)`. Variable nommée `mse` mais c'est du Huber (smooth
L1). Hérité de student où `CAE_loss` avait une variable `mse` aussi (également misnomée).

✅ **Parity avec student**, mais lectable-hostile. **Piste D11-fix-2** (skip) : renommer `huber_term`
ou `recon_term` pour clarté. Bas ROI.

## (d) `lam = 1e-4` hardcoded externe

Paper silent sur la valeur. Student hardcode `lam = 1e-4` inline (L357). Port :
- Param de la fonction (flexible).
- Valeur hardcoded externe dans `sarl/trainer.py:57` (`CAE_LAMBDA = 1e-4`).
- **Pas dans config yaml** — pas d'override CLI possible pour ablations.

**Piste D11-fix-3** (info, skip) : exposer `cae_lambda` dans `SarlTrainingConfig` si besoin
ablation. Hors scope repro Phase F.

## (e) Call sites verification

| Caller                                | Signature usage                          | Status |
|:--------------------------------------|:-----------------------------------------|:------:|
| `sarl/trainer.py:188`                 | `cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)` | ✅    |
| `sarl_cl/trainer.py:283`              | `cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)` (non-CL branch) | ✅    |
| `sarl_cl/trainer.py:302`              | idem (CL branch via `loss_weighter`)     | ✅    |

Tous 3 callers : positional, ordre `(W, td_target, q_s_a, h1, lam)` cohérent avec signature port.

## Fixes identifiées D.11

| ID        | Fix                                                                                   | Scope                             | Effort |
|:----------|---------------------------------------------------------------------------------------|-----------------------------------|:------:|
| D11-fix-1 | Correction docstring function-level L68-69 (contradict module-level post-C.10)        | `sarl/losses.py:67-69` docstring  | 2 min  |
| D11-fix-2 | (skip) renommer variable `mse` → `huber_term` pour clarté                             | `sarl/losses.py:88`               | skip   |
| D11-fix-3 | (skip) exposer `cae_lambda` via config yaml pour ablations                            | config + dataclass                | skip   |

## Cross-reference deviations.md / reviews

- **DETTE-2** : doublon `components.cae_loss` vs `sarl.losses.cae_loss` — tracé, Phase H post-repro.
- **D-002** : SimCLR vs CAE — queued Phase D.22b.
- **D-sarl-recon-bias** : **ne concerne pas `losses.py`** (c'est `sarl/model.py` qui ajoute `b_recon`
  au decoder). Pas d'impact ici.
- **D-sarl-gamma**, **D-sarl-alpha-ema**, etc. D.7/D.9 **ne concernent pas `losses.py`**.
- **Aucune nouvelle deviation D.11.**

## Résumé — `sarl/losses.py`

- ✅ **Huber reconstruction** bit-parity student (default δ=1, reduction=mean).
- ✅ **Jacobian regularizer** bit-parity student (3 ops identiques).
- ✅ **Call signature** préservée paper (confusing naming documenté).
- ✅ **Callers** 3 sites, ordre positional cohérent.
- 🚨 **D11-F1 docstring contradict** post-C.10 : function-level L68-69 dit "gradients flow back
  through W" mais module-level L32-41 corrige en "detached by state_dict". Fix trivial.
- ⚠️ **Variable `mse`** misnomée pour Huber — skip parity.
- ⚠️ **`lam` hardcoded externe** — skip (Phase H).
- **0 nouvelle deviation, 1 fix docstring mineur.**

**D.11 clôturée. 1 fix docstring queued pour next batch. 0 code touché.**
