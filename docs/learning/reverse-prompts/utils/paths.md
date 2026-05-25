# Reverse-prompt — `src/maps/utils/paths.py`

**Module path actuel (sur main) :** `src/maps/utils/paths.py`
**Taille :** ~90 lignes, 1 dataclass + 1 fonction publique (`Paths`,
`get_paths`).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `paths.py` qui centralise la résolution des
> chemins filesystem (data, outputs, models, logs, figures, reports,
> scratch_root) à partir de `config/paths.yaml`. **Single source of
> truth** pour "où ça vit dans le projet".
>
> API publique :
> ```python
> @dataclass(frozen=True)
> class Paths:
>     root, data, outputs, models, logs, figures, reports, scratch_root
>     def ensure_dirs() -> None  # mkdir each owned path
>
> def get_paths(*, root=None) -> Paths:
>     ...
> ```
>
> Project root discovery (ordre) :
> 1. `MAPS_ROOT` env var
> 2. Walk up depuis cwd jusqu'à `config/paths.yaml`
> 3. Fallback `cwd`
>
> SCRATCH override : sur DRAC clusters, `$SCRATCH` est un path
> per-user fast storage. Si set, override `scratch_root` (mais ne mkdir
> jamais — ce path pré-existe et n'est pas à nous).
>
> Resolve les `${.root}/<subpath>` interpolations manuellement (sans
> OmegaConf.resolve car on substitue `root` per-call pour tests).
>
> `ensure_dirs()` crée seulement les paths "owned" (data, outputs,
> models, logs, figures, reports). Ne touche jamais `root` ni
> `scratch_root`.

## 2. Contraintes d'ingénierie

### Pourquoi pas hardcoder ?

Pour 2 raisons :
1. **HPC scratch handoff** : sur Narval/Tamia, `$SCRATCH` est défini
   par SLURM, différent par user. Le code dev local utilise `outputs/`.
   Le même code doit marcher sur les 2 environnements.
2. **Tests** : `tmp_path` fixture pytest doit pouvoir remplacer `root`
   pour isolation. D'où le param `root=None` qui peut être overridden.

### Discovery cwd-based + env var override

Pattern défensif :
- `MAPS_ROOT` env var (explicit override, prend précédence)
- Walk up depuis cwd (auto-discover, marche depuis subdirs)
- cwd fallback (dégradé mais ne crash pas)

Le walk-up cherche `config/paths.yaml` (pas `config/maps.yaml` comme
`config.py`). Petit divergence à harmoniser.

### `@dataclass(frozen=True)`

Frozen pour éviter mutation accidentelle. `Paths` est conceptuellement
immutable une fois résolu — c'est une description statique du
filesystem, pas une config dynamique.

### Field `_owned` privé

Liste des fields qu'on possède (qu'on mkdir). `root` et `scratch_root`
sont exclus parce qu'ils pré-existent et ne sont pas à nous.

## 3. Deviations / dettes présentes

Pas de déviation paper.

**Pas de DETTE active.** Module propre.

Possible amélioration : le discovery walk-up cherche
`config/paths.yaml`, mais `config.py` cherche `config/maps.yaml`. À
harmoniser (l'un ou l'autre comme "ancre" du projet).

## 4. Questions ouvertes

- **Q1 :** SCRATCH override est correct ? Sur Narval `$SCRATCH` n'est
  pas un projet root mais un fast storage area dédié à l'utilisateur.
  Donc on save les outputs dans `$SCRATCH/maps/outputs/` plutôt que
  `outputs/`. C'est ce que fait ce module ? À vérifier.
- **Q2 :** *(à toi)* — qu'est-ce qui te paraît surprenant ?

## 5. Pour le rewrite (Sprint 11+)

### À garder
- API `Paths` dataclass + `get_paths()`
- 3-tier discovery (env, walk-up, cwd)
- SCRATCH override
- `ensure_dirs()` qui ne touche pas root/scratch
- Frozen dataclass

### À améliorer
- **Harmoniser ancre discovery** : choisir `config/maps.yaml` OU
  `config/paths.yaml` comme ancre dans les 2 modules (config.py +
  paths.py), pas un de chaque.
- **Properties pour subpaths fréquents** : `paths.outputs_for(domain,
  seed)` retourne `outputs/<domain>/seed_<seed>/`. Évite la
  duplication dans les trainers.
- **Helper `paths.scratch_or_outputs()`** : retourne `scratch_root` si
  HPC, sinon `outputs`. Pattern fréquent.

### Tests à écrire
- `get_paths(root=tmp_path)` honore override
- `MAPS_ROOT` env var prend précédence
- `$SCRATCH` override
- `ensure_dirs()` crée tous les owned paths
- Discovery walk-up fonctionne depuis un subdir

### Connexion
- Utilise `utils.config.load_config("paths", resolve=False)`
- Utilisé par les trainers pour écrire outputs / models / logs
- Utilisé par les CLI pour configurer `log_file`
