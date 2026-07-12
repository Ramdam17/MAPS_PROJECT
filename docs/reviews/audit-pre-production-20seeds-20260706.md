# Audit pré-production — 20 seeds × 1M steps SARL + MARL (Narval)

**Date :** 2026-07-06
**Contexte :** Guillaume demande de relancer 20 seeds SARL + MARL à 1M steps. Audit complet
du code SARL, SARL+CL et MARL avant de dépenser l'allocation. Quatre revues approfondies
en parallèle : SARL, SARL+CL, MARL, composants partagés + infra SLURM.
**Verdict global : ❌ NE PAS LANCER en l'état.** Chaque domaine a au moins un défaut qui
invaliderait les résultats ou brûlerait l'allocation pour rien.

**Aucune modification n'a été appliquée** — ce document liste les constats et les décisions
à prendre (règle du projet : pas de changement sans go explicite).

---

## 1. Blockers transversaux

### T-1 [CRITIQUE] — L'environnement n'est jamais seedé, nulle part

- **SARL / SARL+CL :** MinAtar crée ses propres `np.random.RandomState` internes non seedés
  (sticky actions p=0.1 dans `external/MinAtar/minatar/environment.py:29` + dynamique de jeu,
  ex. `environments/breakout.py:28`). `set_all_seeds()` seede le RNG numpy *global*, que
  MinAtar n'utilise pas. `Environment.seed()` existe mais n'est appelé dans **aucun** script
  (`scripts/run_sarl.py:69-84`, `scripts/run_sarl_cl.py:317-318`). L'état RNG env est aussi
  absent des payloads de checkpoint.
- **MARL :** le substrat MeltingPot/dmlab2d tire son propre seed Lua/C++ au build
  (`src/maps/experiments/marl/env.py:441`) ; rien ne lui transmet `--seed`.
- **Conséquence :** « seed 42 » n'identifie pas un run — deux exécutions identiques divergent
  dès le premier pas. Reproductibilité par seed cassée ; resume jamais bit-exact ; viole la
  règle du projet « seed control is mandatory ». (Hérité du code étudiant qui ne seedait rien,
  mais le port revendique un déterminisme qu'il n'a pas.)
- **Fix :** `env.seed(effective_seed)` après `_build_env` (SARL, SARL+CL) ; injecter le seed
  dans `substrate.build` (MARL) ; ajouter les états RNG env aux checkpoints ; logger dans
  `deviations.md`.

### T-2 [CRITIQUE] — Le scheduler par défaut gèle l'apprentissage après ~1 % du run (SARL + SARL+CL)

- `StepLR(step_size=1, gamma=0.999)` steppé à **chaque update** (`config/training/sarl.yaml:65-68`,
  `config/training/sarl_cl.yaml:87`, step dans `sarl/trainer.py:208-221` et
  `sarl_cl/trainer.py:377-378`). Avec `training_freq=1` : LR < 1 % après ~4 600 updates,
  numériquement 0 bien avant 1M (0.999^1e6 ≈ 1e-434).
- Le code de référence du papier utilise `step_size=1000`
  (`external/paper_reference/sarl_cl_maps.py:1235,1240`). Déjà flaggé `D-sarl-sched-step`
  comme typo suspectée — mais le défaut de production reste 1.
- **Conséquence :** à 1M frames, 99 % du compute entraîne un réseau figé ; tout l'apprentissage
  se fait avant même la fin de l'anneal ε (100k frames).
- **Décision requise (Guillaume) :** step_size 1 vs 1000 — pilote court (1 seed, 100k frames,
  les deux valeurs) ou résolution auprès des auteurs. Logger dans `deviations.md`.

### T-3 [CRITIQUE] — Cascade first-order = no-op mathématique ; settings 2/4/6 ≡ 1/3/5 à ~50× le coût

- Forward first-order déterministe (pas de dropout) → `a(t) = α·f(x) + (1-α)·a(t-1)` avec
  bootstrap `a(0) = f(x)` retourne exactement `f(x)`, valeur **et** gradient, à chaque itération
  (`components/cascade.py:66-67`, `sarl/model.py:95-105`, `sarl_cl/model.py:108-117`).
  Documenté (`D-sarl-cascade-noop`, warning runtime, `cascade_effective_iters_1` dans metrics,
  test de régression) — mais les 50 itérations s'exécutent quand même : 50 forwards par
  sélection d'action, par update policy, par cible TD, par forward teacher.
- Seul le chemin second-order est « vivant » (dropout resamplé par itération — donc moyenne de
  ~50 masques dropout, pas une intégration temporelle depuis un état de repos).
- **Conséquence :** le contraste factoriel cascade-FO est nul a priori ; ~la moitié de la grille
  brûle ~50× de node-hours pour reproduire l'autre moitié.
- **Décision requise (Guillaume) :** (i) ne pas courir 2/4/6 à 1M ; (ii) corriger la
  stochasticité pour rendre la cascade effective ; ou (iii) court-circuiter les itérations
  redondantes (`effective=1`) pour la parité d'affichage à coût égal.

### T-4 [HIGH] — Le plan « 20 seeds × 1M sur Narval » n'existe dans aucun script

- `sarl_array.sh` / `marl_array.sh` / `sarl_phase_f.sh` ciblent **Tamia**
  (`--account=aip-gdumas85`, `--gpus-per-node=h100:4`) → `sbatch` échoue sur Narval (A100,
  `def-gdumas85_gpu`, cf. `sarl_cl_narval.sh`). Seul SARL+CL a un scaffolding Narval (Phase F.4).
- Listes de seeds dupliquées et incohérentes : 10 seeds (`sarl_array.sh:58-77`), 3 seeds
  (`sarl_phase_f.sh`, `marl_array.sh`, `sarl_cl_narval.sh`), copies indépendantes dans
  `aggregate.sh` et `monitor.sh`. Aucun script à 20 seeds. Éditer 4+ fichiers à la main de
  façon cohérente = mode d'échec assuré (agrégation sur le mauvais ensemble de seeds, silencieuse).
- `num_env_steps` MARL = 300k dans `config/training/marl.yaml:68` (pas 1M).
- `CUDA_VISIBLE_DEVICES=0` sous allocation 4-GPU : 3 GPU sur 4 idle par tâche.
- **Fix :** créer `sarl_array_narval.sh` / `marl_array_narval.sh` sur le modèle
  `sarl_cl_narval.sh` ; source unique pour les seeds (fichier `seeds.env` sourcé partout, ou
  découverte depuis l'arborescence pour aggregate/monitor).

---

## 2. Blockers par domaine

### 2.1 MARL — le plus sévère (4 CRITIQUES indépendants)

| # | Sévérité | Constat | Localisation |
|---|----------|---------|--------------|
| M-C1 | CRITIQUE | **Mini-batches RNN temporellement brouillés** : le générateur récurrent aplatit chunk-major (`np.stack` axis=0 → reshape), le forward GRU dé-aplatit time-major (`view(episode_len, batch_num, ...)`). Chaque update PPO traite des séquences mélangées entre chunks et timesteps. Silencieux (masks/hidden brouillés pareil, rien ne crashe). Le repo de référence `on-policy` stack en `axis=1`. | `marl/data.py:303-316` vs `marl/rnn.py:58-62` |
| M-C2 | CRITIQUE | **Les réseaux meta n'agissent jamais** : `collect()` et `ppo_update` n'utilisent que l'actor/critic baseline ; `MAPSActor.forward` a zéro call site ; `actor_meta` est un réseau dupliqué entraîné uniquement par la BCE wager, sans paramètre partagé avec le réseau qui agit. Les 4 bras `meta=True` sont comportementalement ≡ baseline. Contredit `docs/reviews/marl-architecture.md:41-43`. | `marl/runner.py:258-262`, `marl/trainer.py:269-279`, `marl/policy.py:322-340` |
| M-C3 | CRITIQUE | **`critic_meta` jamais entraîné** : la loss « côté critic » est calculée à travers `actor_meta` (avec des états RNN critic projetés par une couche entraînée sur des états actor — sémantiquement incohérent), puis `optimizers.critic_meta.step()` steppe sur des grads vides (no-op). Vestige d'un fix de crash E.17c mal rerouté. | `marl/trainer.py:350-375` |
| M-C4 | CRITIQUE | **Cibles wager désalignées** : tableau `(T*N, 2)` time-major passé tel quel à chaque mini-batch mélangé chunk-major, à chaque epoch. La tête de wager (signal MAPS eq.13-14) est supervisée avec des cibles effectivement aléatoires. Ne « marche » que parce que `num_mini_batch=1` fait coïncider les shapes. | `marl/trainer.py:294-302,359-366`, `marl/runner.py:489-501` |
| M-H1 | HIGH | Substrat MeltingPot jamais seedé (cf. T-1). | `marl/env.py:441` |
| M-H2 | HIGH | **Cascade rollout ≠ cascade training** : au rollout, 50 applications du GRU sur un pas ; au training, 50 passes pleine-séquence chaînées bout-de-chunk → début de passe. Les log-probs recalculés ≠ log-probs stockés même à poids inchangés → ratios PPO biaisés précisément dans les bras cascade. | `marl/policy.py:168-195`, `marl/rnn.py:49-81` |
| M-H3 | HIGH | **Blend cascade no-op dans `RNNLayer` baseline** : `forward` calcule `output_cascade` mais retourne `norm(x)` (GRU brut). Seul `RNNLayerMeta` applique le blend α — et ces réseaux n'agissent jamais (M-C2). Aucun bras n'expose l'accumulation McClelland au comportement ; le facteur « cascade » mesure « 50 micro-steps GRU ». | `marl/rnn.py:112-120` vs `149-157` |
| M-H4 | HIGH | **Troncature traitée comme terminaison** : `done = term or truncated` avec `max_cycles == episode_length` → GAE ne bootstrappe jamais en fin d'épisode alors que les substrats n'ont pas d'état terminal naturel. Biais systématique des cibles de valeur. Ajouter au minimum un assert `max_cycles == episode_length` (sinon fuite d'état RNN entre resets). | `marl/env.py:257-267`, `marl/data.py:163-186` |
| M-M5 | MEDIUM | BatchNorm2d : rollout en eval (running stats) vs ré-évaluation PPO en train (batch stats, 15 updates/épisode sur les mêmes 1000 samples) → autre source de ratio ≠ 1 à l'epoch 0. | `marl/encoder.py:92`, `marl/trainer.py:545-559` |
| M-M6 | MEDIUM | `active_masks` ne reflète jamais l'élimination par zap (Territory) ; step 0 des épisodes ≥ 2 exclu des losses. | `marl/runner.py:326-327` |
| M-M8 | MEDIUM | Wager entraîné en 2 Bernoullis indépendantes (BCEWithLogits sur one-hot) vs softmax 2-unités du papier. Aussi : `config/maps.yaml` dit `n_wager_units: 1` alors que son propre commentaire dit 2 (MARL hardcode 2). | `marl/policy.py:83,104-108` |

**Points solides MARL :** checkpointing excellent (atomique, optimizer + ValueNorm + 4 flux RNG
avec le fix CPU-uint8, guards au resume), instrumentation NaN E.16 exemplaire, provenance
étudiant/papier citée partout.

### 2.2 SARL

| # | Sévérité | Constat | Localisation |
|---|----------|---------|--------------|
| S-C1 | CRITIQUE | **Resume crashe sur GPU** : `torch.load(map_location=cfg.device)` envoie `rng_torch`/`rng_torch_cuda` sur CUDA ; `torch.set_rng_state()` exige un ByteTensor CPU → `RuntimeError`. Le fix `.cpu().byte()` (commit 19ede75) a été appliqué à SARL+CL (`sarl_cl/training_loop.py:553-556`) et MARL (`marl/runner.py:678,686`) mais **jamais porté au SARL DQN**. Scénario : préemption → requeue → `--resume` → crash → boucle jusqu'à épuisement. Le test existant ne charge qu'en `map_location="cpu"` (angle mort). | `sarl/training_loop.py:447,505-507` |
| S-H1 | HIGH | **`sarl_array.sh` a `--requeue` mais ne passe jamais `--resume`** : une cellule préemptée redémarre de la frame 0 (boucle de restarts infinie à 1M steps). ⚠️ Interaction : ce bug masque actuellement S-C1 — corriger H sans C aggrave. | `scripts/slurm/sarl_array.sh:28,102-107` |
| S-H2 | HIGH | **`run_sarl.py` et `aggregate_sarl.py` divergent sur l'arborescence** : écriture dans `scratch_root/maps/outputs/sarl/...`, lecture dans `outputs/sarl/...` — ne coïncident jamais. Échec bruyant, mais scratch Narval purgé à 60 j : risque de perte de données si « on corrigera l'agrégation plus tard ». | `run_sarl.py:206-215` vs `aggregate_sarl.py:128` |
| S-H3 | HIGH | Pas de handler SIGTERM ; checkpoint toutes les 10k updates sérialise le buffer replay complet (~0.5-0.8 GB, tenseurs GPU) ~100 fois par cellule. | `sarl/training_loop.py:176,332-415` |
| S-M1 | MEDIUM | « Bit-exact resume » de la docstring est faux : épisode en cours abandonné au resume, état env absent. Dégrader la claim, logger les points de resume dans metrics.json. | `training_loop.py:349-364,615-618` |
| S-M2 | MEDIUM | Guards de resume : seulement game/seed/meta/cascade/num_frames — un edit YAML mi-campagne + `--resume` mixe silencieusement deux régimes. | `training_loop.py:322-329` |
| S-M3 | MEDIUM | `aggregate_sarl.py` : aucune validation de provenance (num_frames/seed/game vs répertoire) — une cellule smoke-test 50k oubliée entre dans l'agrégat 1M. z-score = (mean_s − mean_1)/std_1 sans CI ni test (règle labo : CI obligatoire). Métrique = retours ε-greedy d'entraînement, dernier épisode tronqué inclus — à confirmer vs `experiment_matrix.md`. | `aggregate_sarl.py:38-89` |
| S-M4 | MEDIUM | Constantes verrouillées re-hardcodées hors config : 50 itérations dans `_SETTING_TABLE` (`training_loop.py:863-871`), `CAE_LAMBDA=1e-4` (`trainer.py:56`), unités wager, schedule ε (100k frames non rescalé pour 1M — probablement voulu, à documenter). `config/maps.yaml` jamais lu sur le chemin SARL. | divers |
| S-M5 | MEDIUM | Aucun guard NaN sur 1M steps : divergence silencieuse → budget complet brûlé à produire des NaN. | `training_loop.py:675-678` |

### 2.3 SARL+CL

| # | Sévérité | Constat | Localisation |
|---|----------|---------|--------------|
| CL-C1 | CRITIQUE | Scheduler step_size=1 (cf. T-2) — la référence utilise 1000. | `config/training/sarl_cl.yaml:87` |
| CL-C2 | CRITIQUE | Env MinAtar non seedé + état env absent du checkpoint (cf. T-1). | `run_sarl_cl.py:317-318` |
| CL-H1 | HIGH | **Déviation non documentée — target network** : le port utilise un vrai target net figé (sync/500 updates) ; la référence bootstrappe depuis le réseau **online** (tous les call sites passent `policy_net` comme `target_net`, ref:610,694,1488-1505 ; le target_net global de la ref est du code mort). Le port a sans doute « raison » (DQN standard), mais propriétés de stabilité matériellement différentes → doit être une déviation loggée + flag config si la parité est l'objectif. | `sarl_cl/trainer.py:286-290` vs `external/paper_reference/sarl_cl_maps.py` |
| CL-H2 | HIGH | **Hand-off de curriculum incomplet vs référence** : (a) target_net non repris du checkpoint du stage précédent (adapnet) ; (b) état Adam de optimizer2 non repris (`curriculum and meta`) ; (c) `--curriculum` sans `--adaptive` = expérience différente de la référence, permise silencieusement (la prod passe toujours `--adaptive`, donc OK, mais fail-fast souhaitable). | `sarl_cl/training_loop.py:284-316` vs ref:1281-1333 |
| CL-H3 | HIGH | **Aucune évaluation du catastrophic forgetting** : la validation ne tourne que sur l'env courant ; rien n'évalue le policy stage-N sur les jeux 1..N-1. Rattrapable post-hoc depuis les checkpoints de fin de stage (shapes stage-invariantes) mais pas en trajectoire intra-stage. **Vérifier ce que la Figure 7 exige exactement avant lancement.** | `training_loop.py:877-896` |
| CL-H4 | HIGH | Chaîne curriculum : `--time=08:00:00` pour 4 stages séquentiels intenable pour les settings cascade (20-40× plus lents) ; `--requeue` ne couvre pas TIMEOUT → mort silencieuse. Ajouter `--signal=B:USR1@600` + trap, ou chaînage `--dependency=afterany`. | `sarl_cl_curriculum_narval.sh:30` |
| CL-M1 | MEDIUM | Défauts des dataclasses (0.3/0.6/0.1, « Table 11 ») divergent du YAML (0.4/0.4/0.2, décision 2026-05-19 Figure 7) ; docstrings stales. | `sarl_cl/trainer.py:101-103`, `training_loop.py:135-137` |
| CL-M2 | MEDIUM | `metrics.json` n'enregistre ni les poids de loss disputés ni gamma/lr/scheduler — la provenance ne vit que dans `checkpoint.pt` (0.5-1 GB, premier fichier purgé du scratch). Dumper `asdict(cfg)` dans metrics.json. | `training_loop.py:973-1001` |
| CL-M3 | MEDIUM | Buffer replay picklé en ~400k tenseurs CUDA, ~100 réécritures de 0.5-1 GB par cellule → ~5-8 TB de trafic Lustre sur l'array de 90 cellules. | `training_loop.py:443-458` |
| CL-M4 | MEDIUM | `logs/slurm/` doit exister avant `sbatch` (`#SBATCH --output` s'ouvre avant le `mkdir` du script) sinon tout l'array meurt sans logs. | tous les `.sh` |
| CL-M5 | MEDIUM | ε re-anneale 1.0→~0.145 à chaque stage de 100k (jamais le floor 0.1, ε moyen ~0.57) — conforme à la référence, mais à confirmer vs Figure 7. | `sarl/rollout.py:55` |

**Points solides SARL+CL :** le fix RNG `.cpu().byte()` (19ede75) est correct et complet côté
torch ; teachers gelés de façon étanche (`eval()` + `requires_grad_(False)` + `no_grad` +
state_dicts persistés) ; checkpointing atomique avec format_version et guards.

---

## 3. Findings partagés / infra (rapport composants)

- `PYTHONHASHSEED` posé à runtime = no-op pour le process courant (lu au démarrage de
  l'interpréteur) ; le test cimente la fausse garantie. Le poser dans `common.sh` / scripts
  SLURM. Envisager `CUBLAS_WORKSPACE_CONFIG=:4096:8`. (`utils/seeding.py:56`)
- Fallbacks silencieux de config : `getattr(cfg.training, "gamma", 0.999)` etc. dans
  `run_sarl.py:107-116` — une typo YAML est ignorée sans trace. `OmegaConf.merge` des overrides
  hors struct-mode : `-o training.batchsize=64` crée une clé ignorée sans warning
  (`utils/config.py:157`). → `set_struct(True)` + warning sur fallback.
- Cascade au forward target-net SARL : graphe autograd construit pour 50 forwards dont seul
  `detach()` est consommé — `torch.no_grad()` serait numériquement identique et libèrerait
  mémoire/temps sur le chemin de coût dominant (`sarl/trainer.py:181-185`).
- Gradient second-order → first-order **non détaché, volontaire** (parité papier, ordre ①-⑤
  documenté) : le facteur « meta » du factoriel = architecture + interférence de gradient
  conjointement. À garder en tête pour l'interprétation.
- `energy_tracker.py` : dormant (aucun import en prod) mais dangereux si recâblé —
  `subprocess.run` sans timeout, listes non bornées, division par zéro en fin de run.
- Trous de tests : aucun test `map_location="cuda"` au restore (d'où S-C1), aucun test
  end-to-end de reproductibilité d'un entry point (aurait attrapé T-1), aucun test du décode
  TASK_ID des scripts SLURM, `first_order_mlp.py` sans test direct.
- Vérifiés corrects : maths cascade (forme McClelland conforme docstring), comparateur (assert
  de shape anti-broadcast), mapping seed→task bijectif dans tous les scripts array,
  `set_all_seeds` appelé aux 5 entry points, atomicité des checkpoints, direction rsync
  d'`aggregate.sh`.

---

## 4. Ordre de bataille proposé

### Étape 1 — Décisions à prendre avec Guillaume (aucun code)
1. **Scheduler `step_size` : 1 vs 1000** (T-2) — pilote 100k frames × 2 valeurs, ou trancher
   depuis la référence (`1000`). → `deviations.md`.
2. **Sort des bras cascade-FO no-op** (T-3) : skip 2/4/6, court-circuit, ou fix stochasticité.
3. **Sémantique meta MARL** (M-C2) : router `actor_meta`/`critic_meta` au rollout (fidèle au
   design étudiant documenté) vs sonde de wagering passive (lecture Pasquali & Cleeremans) —
   dans ce cas la claim « le meta change les rewards » n'est pas testable par ce port.
4. **Figure 7** : quelle quantité exacte ? (courbes de forgetting intra-stage → il faut ajouter
   l'éval multi-jeux à la boucle AVANT de lancer ; perf finale par stage → post-hoc possible.)
5. **Métrique SARL** : retours ε-greedy d'entraînement (actuel) vs validations greedy.

### Étape 2 — Fixes mécaniques sans ambiguïté (après go)
- S-C1 : porter `.cpu().byte()` dans `sarl/training_loop.py:505-507` + test GPU-restore.
- T-1 : `env.seed()` partout + états RNG env dans les checkpoints (+ MARL substrat).
- S-H1 : `--resume` dans `sarl_array.sh` (APRÈS S-C1, jamais avant).
- S-H2 : réconcilier les chemins `run_sarl.py` ↔ `aggregate_sarl.py`.
- M-C1 : `np.stack(..., axis=1)` dans `recurrent_generator` + test d'ordre temporel.
- M-C4 : cibles wager stockées dans le buffer et découpées comme `actions`.
- M-C3 : `critic_meta` évalué sur `share_obs` + vrais états critic, ou suppression documentée.
- Guards NaN (SARL, SARL+CL) ; guards de resume élargis ; provenance dans metrics.json.

### Étape 3 — Infra Narval
- `sarl_array_narval.sh` / `marl_array_narval.sh` sur le modèle `sarl_cl_narval.sh`
  (`def-gdumas85_gpu`, `--gres=gpu:a100:1`, 1 GPU/tâche).
- Seeds paramétrés une seule fois (20 seeds) ; `aggregate`/`monitor` découvrent depuis l'arbre.
- `--signal=B:USR1@600` + trap pour les timeouts ; `mkdir -p logs/slurm` côté soumission ;
  `num_env_steps` MARL → 1M ; cadence de checkpoint temporelle.

### Étape 4 — Re-validation avant soumission
- Smoke MARL 3 seeds × 300k re-ancré vs papier après fixes (les ratios PPO epoch-0 doivent
  être ≈ 1 hors bras cascade — les logger).
- Pilote SARL 100k × {step_size 1, 1000}.
- Un round-trip checkpoint/resume complet sur GPU Narval par domaine.
- Chronométrer une écriture de checkpoint et une cellule setting-6 (dimensionner `--time`).

---

## 5. Ce qui est solide (les 4 rapports convergent)

- Discipline de provenance exemplaire : `deviations.md` cross-référencé dans docstrings,
  configs, warnings runtime et metrics ; le no-op cascade est même tracé par run.
- Maths cascade/comparateur conformes ; teachers correctement gelés ; checkpoints atomiques
  (tmp + `os.replace`) avec versioning et guards.
- Le checkpointing MARL est le meilleur des trois (RNG complet, ValueNorm, EMA, guards).
- Le problème n'est pas la qualité du port — c'est un ensemble de trous précis, presque tous
  identifiables à une ligne près.

---

*Rapports sources : 4 audits parallèles (SARL, SARL+CL, MARL, composants partagés) du
2026-07-06. Aucun fichier de code modifié.*
