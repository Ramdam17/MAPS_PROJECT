# Reverse-prompt — `src/maps/domains/marl/setting.py`

**Module path actuel :** `src/maps/experiments/marl/setting.py`
**Taille :** ~80 lignes.
**Paper :** §B.4 Table 12, settings 1-6.

---

## 1. Quelle aurait été la spec ?

> Écris une dataclass `MarlSetting` pour la factorial grid 6-cell
> MARL. Mirror `BlindsightSetting` et `AGLSetting` structurellement
> mais avec des nom de fields slightly different (legacy naming
> student `cascade_iterations1` au lieu de `cascade_1st`).
>
> ```python
> @dataclass(frozen=True)
> class MarlSetting:
>     id: str            # "baseline", "maps", etc.
>     label: str
>     meta: bool          # True → MAPSActor/MAPSCritic ; False → MAPPO
>     cascade_iterations1: int  # 1 = off, 50 = on
>     cascade_iterations2: int  # 1 = off, 50 = on
>
>     @classmethod from_dict(d) -> MarlSetting
>     @property has_cascade_1st -> bool  # iter1 > 1
>     @property has_cascade_2nd -> bool  # iter2 > 1
> ```
>
> **6 settings canoniques** (config/experiments/factorial_marl.yaml) :
> 1. `baseline` — meta=False, cascade1=1, cascade2=1
> 2. `cascade_1st_no_meta` — meta=False, cascade1=50, cascade2=1
> 3. `meta_no_cascade` — meta=True, cascade1=1, cascade2=1
> 4. `maps` — meta=True, cascade1=50, cascade2=1 (paper §B.4 simple
>    MAPS)
> 5. `meta_cascade_2nd` — meta=True, cascade1=1, cascade2=50
> 6. `meta_cascade_both` — meta=True, cascade1=50, cascade2=50
>
> Mirror student `train_meltingpot.py:112-142` mapping.

## 2. Contraintes scientifiques

### Paper §B.4 cascade-not-implemented claim

Paper Table 12 preamble dit "cascade non-implémenté pour MARL" (paper
recognized limitation). Mais le student code utilise cascade=50 dans
settings 2/4/6. Notre port supporte les 2 (paper-as-written = no
cascade ; student code = with cascade).

Default factorial = student code values (cascade=50). Override
`cascade_iterations1=1` pour paper-literal.

### Naming legacy (vs Blindsight/AGL)

`cascade_iterations1` (MARL) vs `cascade_1st` (Blindsight/AGL) :
naming différent par hasard (student MARL conv vs MAPS perceptual
conv). À uniformiser Sprint 11+.

## 3. Contraintes d'ingénierie

### `@dataclass(frozen=True)`

Immutable — settings sont des constants, pas mutables.

### `from_dict` classmethod

Permet de construire depuis OmegaConf DictConfig YAML.

### `has_cascade_*` properties

Helper booleans. Cleaner que `setting.cascade_iterations1 > 1`
partout.

## 4. Deviations / dettes présentes

### D-marl-cascade-not-implemented (paper-admitted)

Settings 2/4/6 utilisent cascade=50 (student) vs paper-implicit
no-cascade. Doc only.

### Naming inconsistency cross-domain

`cascade_iterations1` vs `cascade_1st` — différent. Sprint 11+
unification possible mais break parity tests.

### Pas de DETTE active code-wise

## 5. Questions ouvertes

- **Q1 :** Uniformiser naming cross-domain ? Coût : break parity test
  signatures. Bénéfice : DX cohérent.

- **Q2 :** Le 6e setting (`meta_cascade_both`) est paper Setting 6
  full MAPS. Vs setting 4 `maps` (simple MAPS). Différence Phase F.4
  ?

## 6. Pour le rewrite (Sprint 11+)

### À garder
- Frozen dataclass
- 6 canonical settings
- `from_dict` classmethod
- Helper properties

### À améliorer
- **Naming uniformisation** : `cascade_iterations1` →
  `cascade_iterations_1` (with underscore — match Python convention)
- **Validation** : raise si cascade_iter > 100 (probable bug)

### Tests à écrire
- 6 settings construct correctly
- `from_dict` parses YAML
- Properties match
- Immutable (try to mutate raise)

### Connexion
- Utilisé par : `runner.MeltingpotRunner`, `policy.MAPPOPolicy` (use_meta=setting.meta)
- Configure via : `config/experiments/factorial_marl.yaml`
