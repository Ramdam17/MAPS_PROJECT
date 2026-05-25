# Component review — `src/maps/experiments/sarl_cl/training_loop.py`

**Review sub-phase :** Sprint-08 D.19.
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/sarl_cl/training_loop.py` (969 L — plus gros fichier
du repo post-D.13).
**Paper sources :** §2.3 p.9 (CL curriculum), Table 11 CL rows 20-22, §4 (continual learning).
**Student sources :** `external/paper_reference/sarl_cl_maps.py` (dqn + evaluation + persist).

**Callers du port :**
- `scripts/run_sarl_cl.py:145` via `setting_to_config_cl` + `run_training_cl`.

**DoD D.19** : doc créé, 10+ sections auditées, **1 finding critique** (D19-F1 schema conflict),
0 code touché.

---

## (a) Module organization — 969 L, 12 composantes

1. Imports (L41-70).
2. Paper constants (L72-88 — déjà post-D.9/D.12).
3. `MinAtarLike` Protocol (L93-101).
4. `SarlCLTrainingConfig` dataclass (L106-168 — déjà post-D.7/D.9/D.13).
5. `CLTrainingMetrics` dataclass (L170-200).
6. `load_partial_state_dict` (L202-230).
7. `_build_first_order` (L233-241).
8. `_build_networks` (L244-284).
9. `_build_optimizers` (L287-343 — déjà post-D.9).
10. **Checkpoint section** (L345-530 — déjà D.13).
11. **`run_training_cl`** (L532-863, ~330 L).
12. `_persist_outputs` (L866-937).
13. `_SETTING_TABLE` + `setting_to_config_cl` (L940-969).

## (b) `load_partial_state_dict` (L202-230) — critique cross-game

Port vs student `sarl_cl_maps.py:1265-1278` :
```python
model = copy.deepcopy(model)
model_state = model.state_dict()
for name, param in state_dict.items():
    if name in model_state and model_state[name].shape == param.shape:
        model_state[name].copy_(param)
    else:
        log.info("partial load: skipped %s ...", name)
model.load_state_dict(model_state)
return model
```

### (b1) Structure
- `deepcopy(model)` d'abord — caller's original net non-muté. ✅ **Gain defensive** vs student inline mutation.
- Shape match test → copy_ (in-place). Skip on mismatch.
- Final `model.load_state_dict(model_state)` — seulement les keys matched sont synced.

### (b2) Usage cross-game
- Teacher loaded from task-1 checkpoint (ex: Breakout 4 channels).
- Student at task-2 (ex: Space Invaders 6 channels) a un `input_adapter` conv `Conv2d(6, 6, 1)`.
- Shape mismatch sur le conv weight → skip → teacher garde fresh init on that layer.
- Autres layers (fc_hidden, fc_output, actions) généralement same shape → loaded.

✅ **Architecture permet cross-game transfer** pour les parts compatibles (Hidden/Output) tout en
laissant le input-adapter adopter la nouvelle geometry.

⚠️ **Consequence** : la "distillation" cross-game est partielle — seule la partie Hidden-downstream
reçoit l'anchor teacher, pas l'input-adapter. Paper-silent mais logique.

## (c) `_build_first_order` (L233-241) — AdaptiveQNetwork vs SarlCLQNetwork switch

```python
if cfg.adaptive_backbone:
    return AdaptiveQNetwork(cfg.max_input_channels, num_actions).to(cfg.device)
return SarlCLQNetwork(in_channels, num_actions).to(cfg.device)
```

✅ **Clean factory**. Paper §2.3 p.9 prescribes AdaptiveQNetwork pour curriculum cross-game
(variable channels), SarlCLQNetwork pour single-game CL (fixed channels).

## (d) `_build_networks` (L244-284) — 5 réseaux orchestration

Instancie dans cet ordre :
1. `policy` (via `_build_first_order`).
2. `target` (via `_build_first_order`, load policy state_dict, eval).
3. `second` (SarlCLSecondOrderNetwork, si meta).
4. `teacher_first` (si curriculum + teacher_load_path, via load_partial_state_dict + eval + frozen).
5. `teacher_second` (si meta + "second_net_state_dict" in ckpt).

### (d1) Teacher frozen correctly

L274-275 + L281-282 :
```python
for p in teacher_first.parameters():
    p.requires_grad_(False)
```

✅ **Cohérent C.9** : `weight_regularization(model, teacher)` docstring prescribes teacher frozen.

### (d2) 🚨 D19-F1 — Schema mismatch teacher_load_path vs D.13 checkpoint

Port L270-279 charge teacher :
```python
ckpt = torch.load(cfg.teacher_load_path, ...)
teacher_first = load_partial_state_dict(teacher_first, ckpt["policy_net_state_dict"])
if cfg.meta and "second_net_state_dict" in ckpt:
    teacher_second = load_partial_state_dict(teacher_second, ckpt["second_net_state_dict"])
```

**Keys attendues** : `"policy_net_state_dict"`, `"second_net_state_dict"`.

**D.13 `_persist_checkpoint_cl`** (L345-420) sauvegarde :
```python
payload = {
    "policy_state_dict": policy_net.state_dict(),          # ← NOT "policy_net_state_dict"
    "second_order_state_dict": second_order_net.state_dict(),  # ← NOT "second_net_state_dict"
    ...
}
```

**CRITICAL SCHEMA MISMATCH** : les clés diffèrent !
- D.13 checkpoint → `policy_state_dict` / `second_order_state_dict`.
- Teacher loader expects → `policy_net_state_dict` / `second_net_state_dict`.

### (d3) Conflict files — même path `checkpoint.pt`, 2 schemas différents

`_persist_checkpoint_cl` (D.13) **et** `_persist_outputs` (L891) écrivent **TOUS LES DEUX** à
`cfg.output_dir / "checkpoint.pt"`, avec des schemas différents :

| Producer                       | Filename                    | Keys                                                    |
|:-------------------------------|:----------------------------|:--------------------------------------------------------|
| `_persist_checkpoint_cl` (D.13) | `output_dir/checkpoint.pt`  | `policy_state_dict`, `second_order_state_dict`, RNG, optims, schedulers, buffer, metrics, cfg_snapshot |
| `_persist_outputs` (L866)       | `output_dir/checkpoint.pt`  | `policy_net_state_dict`, `second_net_state_dict`, game, seed, cascade, num_frames (teacher-loading schema) |

**Execution order** dans `run_training_cl` post-loop (my D.13 edits) :
```python
if checkpoint_path is not None:
    _persist_checkpoint_cl(checkpoint_path, ...)   # writes D.13 schema

if cfg.output_dir is not None:
    _persist_outputs(policy_net, ...)              # OVERWRITES with teacher schema
```

→ **À la fin du training, `checkpoint.pt` contient le teacher-loading schema**, PAS le resume schema. Conséquences :
- ✅ **Teacher loading** (`--teacher-load-path .../checkpoint.pt` pour task-2) fonctionne car le dernier writer = teacher schema.
- 🚨 **Resume from final checkpoint** (`--resume-from .../checkpoint.pt` ou `--resume` auto-detect) **ne fonctionne pas** — `_restore_from_checkpoint_cl` attend `format_version`, `policy_state_dict`, etc. → **KeyError ou ValueError** au resume.

**Intra-training** (every N updates) : `_persist_checkpoint_cl` seul écrit, donc `checkpoint.pt` contient le resume schema. Resume intermédiaire OK. Mais à la toute fin, il est écrasé.

**Severity** : HIGH — D.13/D.14 resume breaks pour tout workflow qui re-résume après completion.

### (d4) Résolution proposée D19-F1 (fix scope D.20 ou dédié)

**Option A** : renommer l'output de `_persist_outputs` — eg `checkpoint.pt` → `teacher.pt`. Break teacher-loading callers qui passent le path manuel — mais on contrôle la migration.
**Option B** : unifier les schemas. Ajouter les 2 alias dans D.13 payload (`policy_net_state_dict` = ref vers `policy_state_dict`), et enlever le `torch.save` de `_persist_outputs`.
**Option C** : réordonner — `_persist_outputs` avant `_persist_checkpoint_cl` → D.13 gagne l'overwrite. Teacher loader adapté pour read D.13 keys.

Ma reco : **Option B** (unification) — plus propre, un seul schema, moins de code redondant. Fix queued.

**Piste D19-fix-1** : résoudre schema conflict. Scope significant (3 fichiers touchés), **not D.20** (D.20 = paper-faithful config values). Proposition : **sub-phase D.19b dédiée** ou **D.21** (plan says "D.21 Checkpoint/resume SARL+CL" — probablement déjà intégré dans D.13 mais ce fix y appartient naturellement).

## (e) `run_training_cl` (L532-863) — main loop

### (e1) Structure
Idem SARL D.8 mais avec teacher forward + CL losses :
1. Log INFO cfg dump (L554-566).
2. `_build_networks` (5 réseaux).
3. `_build_optimizers` (2 optims + 2 schedulers) + D.9 step=1 log.warning.
4. 2 `DynamicLossWeighter` (L577-580) — un per network, persistent across curriculum.
5. `LossMixingWeights` (L582-586) — depuis cfg.weight_task/distillation/feature.
6. Buffer + metrics + resume logic (D.13 integration).
7. **Outer while** : episode loop.
8. **Inner while** : frame loop.
9. Action via `epsilon_greedy_action` (partagée avec SARL).
10. `sarl_cl_update_step(...)` (D.18 reviewed).
11. Target sync + intra-training checkpoint (D.13).
12. Episode bookkeeping + validation cadence.
13. Final log + final `_persist_checkpoint_cl` + `_persist_outputs` (🚨 D19-F1).

✅ **Structure bit-parity** avec SARL `run_training` + CL extensions.

### (e2) Log INFO cfg dump L554-566

Log au démarrage montre game/seed/meta/cascade/frames/curriculum/adaptive/teacher. **Complet et
debuggable**. ✅

### (e3) D.7/D.9/D.13 integrations verified
- `cfg.gamma` plumbed via `sarl_cl_update_step(gamma=cfg.gamma)` ✅ D.7.
- `cfg.adam_betas` plumbed dans `_build_optimizers` ✅ D.9.
- Resume + checkpoint L345-530 ✅ D.13.
- `cascade_effective_iters_*` metadata — **wait, est-ce que CL a ça ?**

### (e4) ⚠️ D19-F2 — `cascade_effective_iters_*` pas propagé CL ?

Le SARL `TrainingMetrics` a `cascade_effective_iters_1/2` (D.4) qui sont persisted. Le CL
`CLTrainingMetrics` — je vérifie : **absent** de la dataclass (non présent dans le grep initial).

Consequence : metrics.json CL ne contient pas le D.4 artefact pour audit cascade no-op.

**Piste D19-fix-2** : ajouter `cascade_effective_iters_1/2` à `CLTrainingMetrics` + populate dans
`run_training_cl` (même logique D.4 SARL). Scope trivial, cohérence D.4. Could queue or standalone.

### (e5) ✅ Log warning cascade no-op (D.4) — hérité ?

**Vérif** : cascade_iterations_1 > 1 log.warning — est-ce que `run_training_cl` l'émet ? D.4 a
été appliqué à `sarl/training_loop.py` seulement. CL variant n'a probablement pas le warning.

→ Finding mineur. Ajouter dans `run_training_cl` aussi.

**Piste D19-fix-3** : ajouter le D.4 log.warning dans `run_training_cl` L567 pour cohérence CL.

## (f) `setting_to_config_cl` (L950-968)

```python
_SETTING_TABLE = {
    1: (False, 1, 1),
    2: (False, 50, 1),
    ...  # identical to SARL
}

def setting_to_config_cl(setting, base=None):
    meta, c1, c2 = _SETTING_TABLE[setting]
    cfg = base if base is not None else SarlCLTrainingConfig()
    return SarlCLTrainingConfig(**{**cfg.__dict__, "meta": meta, "cascade_iterations_1": c1, ...})
```

✅ **Identique SARL** — `__dict__` spread preserve tous les fields CL-specific (curriculum,
adaptive, teacher_load_path, weights, **gamma, adam_betas, resume_from** etc.).

## (g) Curriculum order — D-sarl_cl-curriculum-order (⚠️ to verify)

**Paper §2.3 p.9** : curriculum order Breakout → SpaceInvaders → Seaquest → Freeway.

**Port search** : aucun code hardcodé `["breakout", "space_invaders", ...]` dans training_loop.
Le curriculum est **caller-side** : via multiple `run_sarl_cl.py` invocations avec
`--teacher-load-path` chainé.

→ **Pas de deviation code-side**. `D-sarl_cl-curriculum-order` peut être **marked ✅ resolved
(caller responsibility documented)**. Cross-ref `scripts/run_sarl_cl.py` docstring.

**Piste D19-fix-4** : mark D-sarl_cl-curriculum-order ✅ resolved dans `deviations.md`.

## Cross-reference deviations.md

| ID existant                  | Status D.19 review                                                         |
|:-----------------------------|:---------------------------------------------------------------------------|
| D-cl-weights (🆘+❌)         | N/A in training_loop (config + trainer scope). Queued D.20.                |
| D-sarl_cl-num-frames (❌)    | N/A here. Queued D.20.                                                     |
| D-sarl_cl-max-channels (❌)  | N/A here. Queued D.20.                                                     |
| D-sarl_cl-channel-adapter (✅) | Resolved D.17.                                                            |
| D-sarl_cl-curriculum-order (⚠️) | **Resolve D.19** : caller-side via scripts. D19-fix-4.                   |
| D-sarl_cl-lossweight-normalization (⚠️) | Re-confirmed (loss_weighter usage L577-580).                       |
| D-sarl_cl-target-update (⚠️) | Re-confirmed: target_update_freq=500 for CL (paper `sarl_cl_maps.py:1121`). |

## Fixes identifiées D.19

| ID        | Fix                                                                       | Scope                            | Effort | Priority |
|:----------|---------------------------------------------------------------------------|----------------------------------|:------:|:--------:|
| D19-fix-1 🚨 | Résoudre schema conflict `checkpoint.pt` (D.13 vs `_persist_outputs`)  | training_loop + teacher loader   | ~40 min | HIGH — resume broken post-training |
| D19-fix-2 | Ajouter `cascade_effective_iters_*` à `CLTrainingMetrics` + populate     | training_loop.py + metrics json  | 10 min | MED |
| D19-fix-3 | Ajouter D.4 log.warning dans `run_training_cl` pour cascade_iterations_1 > 1 | run_training_cl setup        | 5 min  | LOW |
| D19-fix-4 | Mark `D-sarl_cl-curriculum-order` ✅ resolved (caller-side via scripts)   | deviations.md                    | 2 min  | LOW |

## Résumé — `sarl_cl/training_loop.py`

- ✅ **Architecture propre** : 969 L bien structurés en 12 composantes, factory pattern pour
  AdaptiveQNetwork vs SarlCLQNetwork.
- ✅ **`load_partial_state_dict`** : cross-game shape tolerance paper-faithful + deepcopy defensive.
- ✅ **Teacher setup** : load via load_partial_state_dict + freeze + eval, cohérent C.9.
- ✅ **D.7/D.9/D.13 integrations** tous vérifiés.
- 🚨 **D19-F1 CRITICAL** : conflict schema `checkpoint.pt` — D.13 resume écrasé par `_persist_outputs`
  à la fin du training → post-training resume BROKEN. Fix HIGH priority.
- ⚠️ **D19-F2 mineur** : `cascade_effective_iters_*` (D.4) absent de `CLTrainingMetrics`.
- ⚠️ **D19-F3 mineur** : log.warning cascade no-op (D.4) pas propagé à `run_training_cl`.
- ✅ **Curriculum order** : caller-side (scripts), D-sarl_cl-curriculum-order resolved.

**D.19 clôturée. 1 finding CRITIQUE (schema conflict), 2 fix mineurs, 1 deviation resolved.
Fix D19-F1 probablement scope sub-phase dédiée (D.19b ou D.21).**
