# Sprint 02 — Extract MAPS core components

**Status:** ⚪ planned
**Branch:** `refactor/extract-core`
**Owner:** Rémy Ramadour
**Est. effort:** 3 days
**Depends on:** Sprint 01

---

## Context

The MAPS architecture has 3 reusable building blocks that are currently re-implemented in each experiment domain: the **second-order (comparator + wagering) network**, the **cascade model**, and the associated **losses** (contrastive, BCE wagering, distillation for CL). Extracting them into a clean API unlocks:
- unit tests on the components (impossible today — they live inside 2000-line scripts)
- sharing across domains without copy-paste
- easier sensitivity analyses (α, cascade iterations) later

## Reference source

**Blindsight_TMLR.py** is the cleanest implementation (short-ish, well-commented cells). Use it as the source of truth; reconcile divergences with AGL and SARL afterwards.

Paper equations:
- Eq. 1-3: comparator matrix + wagering
- Eq. 4: contrastive loss
- Eq. 5: BCE wagering loss
- Eq. 6: cascade dynamics, α=0.02, 50 iterations

## Objectives

1. Ship a typed, tested `src/maps.components` API
2. Zero behavior drift from Blindsight reference (same tensors out for same seed)
3. NumPy-style docstrings citing paper equation numbers

## Tasks

### 2.1 — Scaffold
- [ ] `src/maps/components/__init__.py`
- [ ] `src/maps/components/cascade.py`
- [ ] `src/maps/components/second_order.py`
- [ ] `src/maps/components/losses.py`
- [ ] `src/maps/networks/__init__.py`
- [ ] `src/maps/networks/first_order_mlp.py` (shared by Blindsight + AGL)

### 2.2 — `cascade.py`
- [ ] Class `CascadeModule(nn.Module)` wrapping a base layer + α, n_iter, reset policy
- [ ] Eq.6: `a_ir(t) = α·Σ w_ij·a_js(t) + (1-α)·a_ir(t-1)`
- [ ] Invariants to preserve:
    - α=1 collapses to single forward pass
    - α=0 raises (degenerate)
    - `reset_state()` called between samples (default) or every N (configurable)
- [ ] Docstring cites McClelland 1989, Goyal & Bengio 2022

### 2.3 — `second_order.py`
- [ ] `ComparatorMatrix`: produces `C_t = X_t - Ŷ_t^(1)` per eq.1
- [ ] `WageringHead`: 2-unit linear layer with dropout per eq.2-3
- [ ] `SecondOrderNetwork`: composes Comparator + WageringHead; hidden dim 100 default
- [ ] Docstring cites Pasquali & Cleeremans 2010, Koch & Preuschoff 2007

### 2.4 — `losses.py`
- [ ] `contrastive_loss(z_i, z_j, temperature)` — eq.4 (Chen et al. 2020 SimCLR form)
- [ ] `wagering_bce_loss(logits, labels)` — eq.5 with class-imbalance handling
- [ ] `distillation_loss(student, teacher, features)` — for SARL+CL teacher network (hybrid loss = task + weight reg + feature)
- [ ] All accept `reduction='mean'|'sum'|'none'`

### 2.5 — `first_order_mlp.py`
- [ ] `FirstOrderMLP(input_dim, hidden_dim=40, output_dim, n_layers)` — the backbone used by Blindsight + AGL
- [ ] Default init + optional cascade wrapping

### 2.6 — Behavioral parity test
- [ ] `tests/parity/test_blindsight_reference.py`:
    - Fix seed 42
    - Instantiate old `FirstOrderNetwork + SecondOrderNetwork + cascade` from a snapshot of `Blindsight_TMLR.py`
    - Instantiate new `maps.networks.FirstOrderMLP` + `maps.components.SecondOrderNetwork` + `CascadeModule`
    - Assert `torch.allclose(old_out, new_out, atol=1e-6)`
- [ ] `tests/parity/test_agl_reference.py` — same idea, AGL snapshot

### 2.7 — Dependency injection friendly
- [ ] No `print` statements, no globals, no file I/O
- [ ] All public classes accept `device`, `dtype` kwargs

## Definition of Done

- `from maps.components import CascadeModule, SecondOrderNetwork` works
- `from maps.components.losses import contrastive_loss, wagering_bce_loss, distillation_loss` works
- `pytest tests/parity` passes (behavioral equivalence with legacy code)
- `pytest tests/unit/components` passes (shape/grad/invariants — see Sprint 05)
- All public symbols have NumPy docstrings with paper eq. references
- `uv run ruff check src/maps/components src/maps/networks` passes

## Risks

- Blindsight and AGL may have subtly different cascade implementations (e.g. different reset policy). Reconcile: if they differ, parametrize `CascadeModule(reset_policy='per_sample'|'per_batch')` and default to Blindsight's.
- Paper α=0.02 is hardcoded in 2 places per domain × 4 domains. Centralising forces a test: `0.02 * 50 = 1.0` is suspicious (equivalent to 1 full pass spread over 50 steps).
