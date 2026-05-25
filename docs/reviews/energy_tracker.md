# Component review — `src/maps/utils/energy_tracker.py`

**Review sub-phase :** Sprint-08 C.14 (standalone — séparé de `utils.md` C.13).
**Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/utils/energy_tracker.py` (612 lignes, 2 classes, 18 méthodes).
**Paper sources :** aucune — le paper ne mesure pas l'énergie (D-marl-energy-not-tracked implicite).
**Student sources :**
- Byte-identical copies (pré-Sprint-01) :
  - `BLINDSIGHT/energy_tracker.py`
  - `AGL/energy_tracker.py`
  - `MARL/MAPPO-ATTENTION/energy_tracker.py`
  - `SARL/MinAtar/examples/energy_tracker.py`
- Dedup en Sprint-01 → unique copie dans `src/maps/utils/energy_tracker.py` (+ `print()→log` migration
  en Sprint-04b 4.9).
**Callers** :
- `src/maps/**/*.py` : **0 caller**.
- `MARL/MAPPO-ATTENTION/onpolicy/runner/separated/meltingpot_runner.py:14` : **1 caller externe**
  (MARL runner, paper reproduction ref, non-actively-maintained pour Phase F reproduction SARL/
  Blindsight/AGL).

**DoD global C.14** : doc créé, architecture auditée, statut dead-code-for-now clarifié, DETTE-4
candidate.

---

## Architecture (C.14-a)

2 classes publiques, ~612 L de code (après print→log migration Sprint-04b).

### `NvidiaEnergyTracker` (L16-441, ~425 L)

Tracker GPU basé sur `subprocess.run(nvidia-smi ...)`. Pattern :

```python
tracker = NvidiaEnergyTracker(project_name="...", country_iso_code="CAN")
tracker.start()                # poll nvidia-smi périodiquement
# ... training ...
result = tracker.stop()        # return dict {energy_kwh, emissions_kg, duration_hours, ...}
```

**Méthodes** (classes listées) :
- `__init__` — check nvidia-smi, torch.cuda.device_count, init readings.
- `_check_nvidia_smi` / `_get_gpu_names` — subprocess probe.
- `start` / `stop` — timers + log readings.
- `log_point` — snapshot immédiat.
- `_log_gpu_metrics` — snapshot nvidia-smi (power, temp, utilization).
- `_calculate_total_energy` — intégration puissance × intervalle.
- `_calculate_power_metrics` — avg/max/min.
- `_save_logs` — CSV + JSON output.
- `_generate_plots` — matplotlib plots.

### `MLModelEnergyEfficiency` (L444-612, ~168 L)

Post-processing wrapper autour de `NvidiaEnergyTracker` :

```python
efficiency = MLModelEnergyEfficiency(tracker, model_name="SARL_seed42")
efficiency.start_tracking()
# ... training ...
results = efficiency.stop_tracking()
metrics = efficiency.calculate_efficiency_metrics(training_results, results)
efficiency.generate_efficiency_report(metrics)
```

Calcule `kwh_per_percent_accuracy`, `accuracy_per_kwh`, `kwh_per_million_params`, etc. Génère
un rapport Markdown.

## (b) Pré-conditions runtime strictes

`NvidiaEnergyTracker.__init__` **raise RuntimeError** si :
1. `nvidia-smi` introuvable (`_check_nvidia_smi` return False).
2. `torch.cuda.device_count() == 0` (pas de GPU).

→ **Instanciation impossible sur dev box macOS/CPU-only Linux.** L'import du module reste safe
(pas de side-effect au module-level).

**Impact** : Phase F runs SARL sur tamia (H100/H200 GPU) → peuvent instancier. Phase F runs
Blindsight/AGL (paper → Blindsight sur RTX3070 8GB, voir Table 9 ; mais notre setup peut varier)
→ si lancés en CPU uniquement, instanciation plante. Aucun problème actuel car **aucun caller
src/** n'invoque ce tracker.

## (c) Import side-effects

Ligne 7 : `import matplotlib.pyplot as plt`.

**Impact** : importer `maps.utils.energy_tracker` **déclenche matplotlib startup** (~200-500 ms
cold), même si on n'utilise que la classe sans plot. Problème mineur mais piège si `energy_tracker`
était ré-importé par le path `from maps.utils import ...` (actuellement non — `__init__.py`
n'exporte pas `energy_tracker`, confirmé en C.13).

**Piste C14-fix-1** : déplacer `import matplotlib.pyplot as plt` en lazy import dans
`_generate_plots` (L389) — 1-ligne change, gain ~300ms cold import.

## (d) DETTE-4 candidate — dead code dans `src/`

**Historique** :
- Sprint-01 (dedup) : déplacé les 4 copies byte-identiques → unique path `src/maps/utils/`.
- Sprint-04b (4.9) : migré 14 `print()` → structured `log.info/warn/error`, scoped ruff ignore
  `T20` retiré.
- **Aucun caller dans `src/`** (grep confirmé 2026-04-20).
- Unique caller externe : `MARL/MAPPO-ATTENTION/onpolicy/runner/separated/meltingpot_runner.py:13-14`
  → `from maps.utils.energy_tracker import NvidiaEnergyTracker, MLModelEnergyEfficiency`.

**Status** :
- MARL (MeltingPot) runner est **gardé dans le repo sous `MARL/`** (paper reproduction ref,
  préservé par la policy "preserve everything that reproduces paper").
- Phase E (port MARL vers `src/maps/experiments/marl/`) pourrait ressortir ce tracker comme
  dépendance. Si MARL port n'est jamais fait (possible selon décisions Phase G), le tracker
  devient vraiment dead code dans `src/`.

**Piste C14-fix-2** : ajouter DETTE-4 dans `deviations.md` :
- **Location** : `src/maps/utils/energy_tracker.py`.
- **Problem** : 612 L, 0 caller dans `src/`, 1 caller externe MARL. Pas reproduction-critical
  (paper ne mesure pas l'énergie).
- **Justification** : gardé car MARL/ peut l'utiliser si/quand Phase E port se fait. Retirer
  maintenant casserait MARL runner qui est paper-reproduction-preserved.
- **Resolution** : Phase H post-reproduction, soit (a) déplacer dans `maps.experiments.marl`
  quand port effectué, soit (b) supprimer si MARL port jamais fait et Phase G accepte.

## (e) Emissions factors hardcoded

L69-79 : 9 pays ISO codes → emission factor (kg CO2/kWh), default 0.47 si inconnu.

**Source** docstring : *"Source: https://www.carbonfootprint.com/international_electricity_factors.html"*.
Valeurs 2023-ish estimées, vieillissent (grid se décarbonise).

**Note** :
- Canada (default caller config `CAN`) = 0.1 kg CO2/kWh — **optimiste** pour Québec (hydro) mais
  pas représentatif Ontario/Alberta.
- France = 0.05 — cohérent (nucléaire).
- Default fallback 0.47 — ~moyenne mondiale 2023.

**Piste C14-fix-3** (optionnel) : sourcer les valeurs via un fichier config YAML pour qu'on puisse
les mettre à jour sans toucher le code. Hors scope, skip C.14.

## (f) Test coverage

**Pytest coverage Sprint-05** : `energy_tracker.py` **omitted** du coverage-fail-under=80%
(documenté dans `docs/sprints/sprint-05-tests-and-ci.md:17`). Raison : pas de GPU en CI.

**Impact** : le module n'a **pas de tests unitaires**. Les comportements suivants ne sont pas
couverts :
- Integration de la puissance × intervalle (calcul kWh).
- Parsing de la sortie `nvidia-smi --query-gpu=...`.
- Génération plots / CSV.

**Piste C14-fix-4** (skip C.14) : ajouter tests mockant `subprocess.run` pour simuler
`nvidia-smi` output. Hors scope repro, à faire Phase H.

## (g) print→log migration (Sprint-04b 4.9) — déjà effectuée

Historique : 14 `print()` → 9 `log.info`, 4 `log.warn`, 1 `log.error` + scoped ruff `T20` ignore
retiré. Conforme CLAUDE.md lab policy (no print in production).

✅ **Déjà fait en Sprint-04b**. Audit confirme : `grep "^\s*print\(" src/maps/utils/energy_tracker.py`
retourne `0 match`. Validé.

## Fixes identifiées C.14

| ID        | Fix                                                            | Scope                                       | Effort |
|:----------|----------------------------------------------------------------|---------------------------------------------|:------:|
| C14-fix-1 | Lazy import matplotlib dans `_generate_plots` (gain ~300ms cold) | `energy_tracker.py:7` + L389            | 5 min  |
| C14-fix-2 | Ajouter DETTE-4 (dead code src/, dep externe MARL) dans `deviations.md` | `deviations.md` DETTE section        | 5 min  |
| C14-fix-3 | (skip) sourcer emissions factors via YAML                      | —                                           | skip   |
| C14-fix-4 | (skip) tests unitaires avec mock subprocess                    | —                                           | skip   |

## Cross-reference deviations.md

- **Aucune nouvelle deviation paper** — le paper ne mesure pas l'énergie.
- **DETTE-4 à ajouter** — tracker dead code dans `src/`, kept pour MARL.

## Résumé — `energy_tracker.py`

- ✅ **Module propre** post print→log migration Sprint-04b.
- ✅ **Pré-conditions strictes** à `__init__` (nvidia-smi + GPU) — safe fallback.
- ⚠️ **Import matplotlib eager** → ~300ms cold start piège → C14-fix-1.
- ⚠️ **0 caller src/** → DETTE-4 candidate (pas orphan car MARL consomme).
- ⚠️ **0 test unit** (omitted du coverage gate CI) → à corriger Phase H.
- ✅ **Non-reproduction-critical** — pas utilisé dans Blindsight/AGL/SARL/SARL+CL training.
- **0 divergence paper.**

**C.14 clôturée. 2 fixes actionnables (C14-fix-1 matplotlib lazy, C14-fix-2 DETTE-4), 2 skips
Phase H. Batch d'application recommandé en C.15 ou C.16 regroupé avec autres fixes utils/.**
