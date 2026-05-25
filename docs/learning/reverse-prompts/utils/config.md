# Reverse-prompt — `src/maps/utils/config.py`

**Module path actuel (sur main) :** `src/maps/utils/config.py`
**Taille :** ~150 lignes, 1 fonction publique (`load_config`).
**Référence externe :** OmegaConf, Hydra (composition mimicked).

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `config.py` qui charge des YAML de config avec
> OmegaConf et supporte une composition style Hydra légère sans pull
> Hydra runtime complet.
>
> Two-level pattern :
> 1. `config/maps.yaml` = constantes scientifiques verrouillées (α
>    cascade, dimensions, wager units, seeds)
> 2. `config/domains/<domain>/training.yaml` = hyperparams training-loop
>    (optimizer, lr, epochs) qui composent depuis maps.yaml via :
>    ```yaml
>    defaults:
>      - /maps@_here_
>    ```
>
> API publique :
> ```python
> load_config(name, *, overrides=None, resolve=True) -> DictConfig
> ```
>
> Where `name` accepte :
> - bare name relatif à `config/` : `"maps"`, `"training/blindsight"`
> - chemin absolu vers un `.yaml`
>
> `overrides` = liste de dot-paths style Hydra
> (`["train.n_epochs=10", "cascade.alpha=0.05"]`).
>
> Project root discovery : walk up depuis cwd jusqu'à trouver
> `config/maps.yaml`. Fallback : cwd. Discovery à module-import time
> (constant `CONFIG_ROOT`).
>
> Recursive `defaults:` resolution : un fichier composé peut lui-même
> composer (chain). Support le suffix `@_here_` Hydra (ignoré, on merge
> toujours au root). Support le sentinel `_self_` (ignoré aussi).
>
> Raise clair (`FileNotFoundError`, `ValueError`) si un default référence
> un fichier manquant ou si le YAML top-level n'est pas un mapping.

## 2. Contraintes scientifiques

### Verrouillage des constantes

`config/maps.yaml` contient les valeurs canoniques du paper (cascade
α=0.02, n_iter=50, first_order hidden_dim=40, second_order dropout=0.5,
wager_units=1, CAE λ=1e-4, distillation T=2.0, seed=42). **Toute
deviation doit être loggée** dans `docs/reproduction/deviations.md`.

Le system de composition garantit que :
- Les domain configs **héritent** des constantes
- Les domain configs peuvent **overrider** ponctuellement (Blindsight
  `first_order.input_dim=100`)
- Les CLI overrides ont la **plus haute précédence**

## 3. Contraintes d'ingénierie

### Pourquoi pas Hydra complet ?

Hydra apporte beaucoup d'ergonomie (multi-run, sweeper, logging
auto) mais aussi beaucoup de complexité (decorator @hydra.main qui
prend over l'entry point, working directory cwd-shifting magic, etc.).

MAPS n'a pas besoin de tout ça : composition statique seulement. Donc
on réimplémente le strict minimum (lecture YAML + merge `defaults:`)
avec OmegaConf comme backend. ~150 lignes au lieu de toute la
dépendance Hydra.

### Discovery cwd-based + caveat

`CONFIG_ROOT` est résolu **à module-import time** en walkant cwd
upward. Si un processus change cwd APRÈS l'import, `CONFIG_ROOT`
pointe toujours sur le root discovered à l'import. Caveat documenté ;
en pratique pas un problème (uv run depuis le repo root).

### Recursivité `_apply_defaults`

Un fichier composé peut composer à son tour. Le code récurse pour
résoudre les chaînes (`paths.yaml` → `maps.yaml` → ...). **Cycle
detection absent** — un cycle silencieux ferait un stack overflow.
Cas hypothétique vu l'usage actuel.

### Hydra compat partielle

- `@_here_` suffix : accepté, ignoré (on merge au root toujours)
- `_self_` sentinel : accepté, ignoré (on applique self après defaults)
- `${...}` interpolations : résolues à l'appel sauf `resolve=False`
- Override syntax `key=value` via `OmegaConf.from_dotlist` : standard

## 4. Deviations / dettes présentes

Pas de déviation paper sur ce module (infra pure).

**Pas de DETTE active** sur le code lui-même. Mais l'architecture
"two-level + composition" est un **choix d'ingénierie** qui pourrait
être interrogé :
- **Pro** : reproductibilité, locking, override CLI propre
- **Con** : complexité conceptuelle pour un newcomer (Hydra-style sans
  Hydra)

## 5. Questions ouvertes

- **Q1 :** OmegaConf vs alternative (pydantic + YAML / dataclasses + yaml.safe_load)
  — choix défendable mais alternatives auraient été plus typées.
  Bénéfice OmegaConf : interpolations `${...}`, dot-access naturel,
  Hydra-compat partiel.
- **Q2 :** Faut-il documenter une schema (pydantic / typed dataclass)
  par-dessus OmegaConf pour avoir validation + autocomplete IDE ? Cost
  significatif mais protège contre les typos.
- **Q3 :** *(à toi)* — qu'est-ce qui te paraît surprenant ?

## 6. Pour le rewrite (Sprint 11+)

### À garder
- API `load_config(name, overrides=None, resolve=True)`
- Composition récursive `defaults:` minimaliste
- Discovery cwd-based avec fallback
- OmegaConf backend

### À améliorer
- **Cycle detection** dans `_apply_defaults` (set de chemins visités)
- **Typed wrapper** : exposer des dataclasses `MapsConfig`,
  `BlindsightConfig`, etc. qui valident la structure (utiliser
  OmegaConf.structured ou pydantic)
- **CLI typer integration** : helper `parse_overrides(*overrides) →
  list[str]` qui valide la syntaxe avant load
- **Caveat cwd** : option pour re-discover si needed

### Tests à écrire
- `load_config("maps")` retourne les bonnes valeurs
- Composition : `load_config("training/blindsight")` a bien les
  constantes maps.yaml + overrides Blindsight
- Override CLI : `load_config("maps", overrides=["cascade.alpha=0.5"])`
- Erreur claire si fichier manquant, si `defaults` malformé
- Cycle (futur) : raise au lieu de stack overflow

### Connexion
- Utilisé par `cli.py` de chaque domaine pour parser configs + CLI
  overrides
- Utilisé par `utils/paths.py` pour charger `paths.yaml`
