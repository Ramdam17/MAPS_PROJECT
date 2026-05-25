# Reverse-prompt — `src/maps/utils/energy_tracker.py`

**Module path actuel (sur main) :** `src/maps/utils/energy_tracker.py`
**Taille :** 618 lignes — gros module mais mécanique (pas de logique
scientifique cœur).
**Référence externe :** `nvidia-smi`, Codecarbon, ISO 3166 country codes
pour facteurs CO2.

---

## 1. Quelle aurait été la spec ?

> Écris un module Python `energy_tracker.py` qui mesure la consommation
> énergétique GPU pendant l'entraînement, calcule les émissions CO2
> selon le pays (facteurs IEA), et produit des plots + CSV pour les
> reports paper. Le module est self-contained — ne dépend pas de
> Codecarbon (alternative plus lourde).
>
> Deux classes :
>
> **1. `NvidiaEnergyTracker`** : sampling périodique de `nvidia-smi
> --query-gpu=power.draw,utilization.gpu,temperature.gpu,memory.used`
> via subprocess. Stocke time series en DataFrame, calcule
> kWh (= Σ power_W · Δt_s / 3600 / 1000), multiplie par le facteur CO2
> du pays (CAN=0.1, FRA=0.05, etc.).
>
> Méthodes : `start()`, `stop()`, `record_step(metrics)`,
> `save_results(output_dir)`, `_generate_plots()` (lazy import
> matplotlib).
>
> **2. `MLModelEnergyEfficiency`** (wrapper sémantique) : combine
> energy + performance metrics pour produire un "energy efficiency
> score" (perf/kWh).
>
> Use case : MARL runs longs (16h+ sur A100). Pas utilisé par
> Blindsight/AGL/SARL/SARL+CL (runs courts, énergie négligeable).
>
> Si nvidia-smi absent ou pas de GPU → raise RuntimeError au
> constructor.

## 2. Contraintes d'ingénierie

### subprocess vs nvml binding

Le code utilise `subprocess.run(["nvidia-smi", "--query-gpu=...",
"--format=csv"])` au lieu de `pynvml` / `nvidia-ml-py`. Pourquoi :
- subprocess marche sans deps installées (Codecarbon style)
- Plus tolérant aux versions de driver NVIDIA
- Moins fragile aux bindings Python qui changent
- Inconvénient : ~10ms overhead par appel subprocess (négligeable au
  sampling 5s default)

### Lazy import matplotlib

`matplotlib.pyplot` est importé **dans** `_generate_plots()` plutôt
qu'en tête de module. Économise ~300ms de cold-import pour les
processes qui ne tracent jamais (la plupart). Standard practice pour
des libs lourdes.

### Country codes IEA

Hardcoded dict `emissions_factors = {"CAN": 0.1, "FRA": 0.05, "USA":
0.38, ...}`. Source : carbonfootprint.com international factors.
Default global average (0.47) si pays non listé. **Outdated probable**
(facteurs CO2 changent annuellement) — à mettre à jour ou exposer en
config YAML.

## 3. Deviations / dettes présentes

### DETTE-4 — Module mal placé

`energy_tracker.py` vit dans `src/maps/utils/` mais a **0 callers
dans src/maps/** ! Le seul consommateur est
`MARL/MAPPO-ATTENTION/onpolicy/runner/separated/meltingpot_runner.py`
(legacy MARL runner, pas porté dans src/maps/experiments/marl/ encore).

Le module survit pour 2 raisons :
1. Le MARL runner legacy l'importe (zone non encore portée)
2. Le paper s'en sert pour le table CO2 footprint (annexe)

**Resolution prévue Sprint 15 (MARL refactor)** : déplacer dans
`src/maps/domains/marl/energy.py` ou similaire, puisque c'est seulement
utilisé là.

### Note sur 14 print() résiduels

Le module avait 14 `print()` originaux (legacy Codecarbon wrapper).
Sprint 04b a migré 9 vers `log.info`, 4 vers `log.warning`, 1 vers
`log.error`. Le scoped ignore T201 a été retiré. ✅ Done.

### Pas d'usage paper-faithful

Aucun chiffre Tables 5-7 ne dépend de ce module. La fidélité
numérique n'est pas un enjeu — c'est de l'infrastructure auxiliaire.

## 4. Questions ouvertes

- **Q1 :** Est-ce qu'on a vraiment besoin de ce module pour reproduire
  les Tables paper ? Si non, pourquoi le maintenir ? Si on le supprime,
  on perd quoi (les CO2 emissions annexes peut-être) ?
- **Q2 :** Codecarbon est-il une meilleure dépendance externe que cette
  réimplémentation ? Codecarbon est maintenu, fait pareil + plus, deps
  `nvidia-ml-py` standard. Trade-off : add deps vs maintain code.

## 5. Pour le rewrite (Sprint 11+)

### À garder ou supprimer ?

**Option A — Garder, déplacer (DETTE-4 resolution path) :**
- Move to `src/maps/domains/marl/energy.py`
- Update import in MARL runner
- Move tests if any (currently no tests)

**Option B — Supprimer, switcher à Codecarbon :**
- Ajouter `codecarbon` dep dans `[project.optional-dependencies]
  energy`
- Réécrire MARL runner pour utiliser `codecarbon.EmissionsTracker`
- Supprimer notre tracker maison

**Option C — Garder en utils/ tel quel :**
- Reconnaître que d'autres domaines pourraient l'utiliser plus tard
- Coût zero d'inactivité

À discuter au Sprint 15. Pour Sprint 11+ (autres domaines), ce module
n'est pas dans le chemin critique.

### Si on garde, à améliorer
- **Country factors en config YAML** : `config/energy/co2_factors.yaml`
- **Tests** : sampler un fake nvidia-smi, vérifier les calculs kWh +
  CO2
- **API plus pythonique** : `with EnergyTracker(...) as t:` contextmanager
- **Async sampling** : current sampling est blocking dans `record_step`,
  pourrait être async thread

### Connexion
- Currently 0 callers in src/maps/ (cf. DETTE-4)
- Imported by `MARL/MAPPO-ATTENTION/onpolicy/runner/separated/
  meltingpot_runner.py` (legacy, hors src/maps/)
- Should become local to `src/maps/domains/marl/` per DETTE-4

## Méta — pour Claude chat

```text
Je travaille sur un module Python qui mesure la consommation énergétique
GPU pendant l'entraînement ML, pour produire des CO2 footprints dans des
papers de recherche. Trois questions :

1. Pour MARL training sur A100 16h+, quelle est la précision attendue
   d'un sampling nvidia-smi à 5s d'intervalle ? Quelle erreur sur le
   total kWh ?
2. Les facteurs CO2 par pays (carbonfootprint.com) — y a-t-il une
   source plus rigoureuse et à jour (IEA, EPA, ADEME) ?
3. Codecarbon vs custom code : qu'est-ce que Codecarbon fait que mon
   custom NvidiaEnergyTracker ne fait pas, et vice-versa ?
```
