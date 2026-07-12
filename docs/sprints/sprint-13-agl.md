# Sprint 13 — AGL rewrite + tests + paper reproduction

**Status:** ✅ done (2026-07-12)
**Branch:** `refactor/agl` (branchée depuis `6f07916` — cœur partagé complet : core + utils + networks/first_order_mlp)
**Owner:** Rémy Ramadour
**Depends on:** Sprint 11 ✅ (core/) + Sprint 12 ✅ (Blindsight, le patron) + shared core à `6f07916`
**Numerical reference:** `external/paper_reference/agl_tmlr.py` (verbatim
Vargas code — produit les chiffres paper Table 5b/5c).

---

## Context

Sprint 12 a livré Blindsight end-to-end (parité 4 niveaux vs paper_reference).
Sprint 13 porte **AGL** (Artificial Grammar Learning, paradigme Dienes 1997) sur
sa propre branche, ne contenant QUE `core/` partagé + `domains/agl/`.

AGL réutilise tel quel : `networks/first_order_mlp.py` (avec
`make_chunked_sigmoid(6)`), tout `utils/`, tout `core/`. Il ajoute
`domains/agl/{data, pool, trainer, cli}.py`.

AGL est plus riche que Blindsight : protocole en 3 phases (pretrain → reset →
20-cellules High/Low awareness), qui est *le* mécanisme de la dissociation
implicite/explicite.

## Goal

AGL **paper-faithful end-to-end**, bit-exact vs `agl_tmlr.py`, reproduisant les
Tables 5b/5c. Reproduction FIDÈLE, bugs compris (pas de correction ; voir D13.2).

---

## Décisions Day-1 (verrouillées 2026-07-12)

### D13.1 — Source de vérité & parité
Reproduire `external/paper_reference/agl_tmlr.py` **verbatim, bit-exact**. Tous les
hyperparamètres découlent de la source, pas du texte du papier.

### D13.2 — Baseline fidèle, PAS les « overrides production » du Sprint-08 ⚠️
On construit la version **fidèle à la source** (bugs compris) :
- `n_epochs_pretrain` = valeur du code source (30 attendu), pas 60 (papier Table 10).
- **Pasquali wager hidden absent** si la source l'a absent (bug étudiant) — on ne le
  restaure PAS par défaut.
- RangerVA + `train_meta_frozen_in_training` = comportement source.
Les « améliorations production » du Sprint-08 D.28 (hidden restauré, epochs=60, etc.)
seront des options **désactivées par défaut**, ajoutées PLUS TARD (navigation commune).
Aucune correction furtive.

### D13.3 — Architecture trainer : refacto classe (calquée Blindsight)
`AGLTrainer` (build / pre_train / training / evaluate / evaluate_pool) +
`AGLNetworkPool`, avec le helper module-level partagé `_run_training_loop`.

### D13.4 — Portée de parité : 4 tiers Blindsight + Tier 5 AGL-spécifique ⚠️
- Tier 1 — data : `Generate_Grammar_A/B`, `Generate_Word_Random`, `encode_word`,
  `target_second` bit-exact (seed fixé).
- Tier 2 — réseau : `FirstOrderMLP` forward + 50-cascade + backward bit-exact.
- Tier 3 — 1 pas d'update (loss + backward + step + scheduler).
- Tier 4-light — `pre_train(n_epochs=2)` losses bit-exact.
- **Tier 5 — AGL-spécifique** : (a) reset first-order bit-exact (state_dict initial
  restauré après pretrain), (b) mini-pool (2 cellules, 1-2 epochs) + `evaluate_pool`
  agrégation par tier High/Low. Sans ça on ne teste pas le cœur d'AGL.

### D13.5 — RangerVA
RangerVA par défaut + fallback ADAMAX avec warning (extra `agl` installé). Conforme
source + ablation A4 (RangerVA essentiel : MAE 0.014 vs ADAMAX 0.038).

---

## Plan de phases (calqué Sprint 12)

| Phase | Contenu | Commit |
|-------|---------|--------|
| 13.A | Ouvrir le sprint + décisions Day-1 | ce commit |
| 13.C | `domains/agl/data.py` (Generate_*, encode_word, Array_Words, target_second) + Tier 1 | |
| 13.D | `domains/agl/trainer.py` (AGLTrainer + `_run_training_loop`) + reset | |
| 13.E | `domains/agl/pool.py` (AGLNetworkPool 20-cellules High/Low) | |
| 13.F | `domains/agl/cli.py` (Typer, 4-phase, output layout) | |
| 13.G | Parité 4 tiers + Tier 5 (reset + mini-pool) vs paper_reference | |
| 13.H | walkthrough + closeout | |

## Sources & specs (déjà présentes)
- Source : `external/paper_reference/agl_tmlr.py` (Generate_* L268-351, encode_word
  L361, Array_Words L392, target_second L421, create_networks L814, pre_train L619,
  reset L751).
- Specs : `docs/learning/reverse-prompts/domains/agl/{data,pool,trainer,cli}.md`.
- Risque clé : **D-agl-reset** (L751 `load_state_dict(initial_first_order_weights)`)
  — le mécanisme de la dissociation. Parité Tier 5 obligatoire.

## Dette / findings d'audit
AGL n'était PAS dans l'audit pré-production (qui couvrait SARL/MARL). Aucune dette
d'audit à replier ici. Les déviations connues (D.28.a-i) sont documentées dans les
specs et **reproduites** telles quelles (D13.2) — non corrigées.

---

## Closeout (2026-07-12)

Toutes les phases livrées sur `refactor/agl` (rien poussé). Un commit par phase.

| Phase | Commit | Contenu | Tests |
|-------|--------|---------|-------|
| 13.A | `5b6bc22` | ouvrir sprint + 5 décisions Day-1 | — |
| 13.C | `30ee56d` | `data.py` (grammaires FSM, encode, wager) | Tier-1 9/9 bit-exact |
| 13.D | `c62fee5` | `trainer.py` (pre-train + reset + Grammar-A + `_run_training_loop`) | 8 unit |
| 13.E | `03e4ca2` | `pool.py` (20-cellules High/Low + evaluate) | 5 unit |
| 13.F | `15ba4ac` | `cli.py` (pipeline 4-phases) | 2 intégration |
| 13.G | `472652c` | parité Tier-2/4/5 | 7 (réseau + boucle + reset) |

**Total : 31 tests AGL verts** (9 parité data + 7 parité tiers + 8 trainer + 5 pool + 2 CLI).

### Mécaniques load-bearing reproduites (fidèles à la source, D13.2)
1. **D-agl-reset** (`trainer.py` build cache + `pre_train` L751-equiv) : le 1st-order est
   remis à ses poids initiaux après le pretrain. Vérifié **bit-exact** (Tier 5). C'est *le*
   levier de la dissociation High/Low awareness.
2. **Two-loss gradient pattern** : `optimizer_1.zero_grad()` avant
   `loss_2.backward(retain_graph=True)`, puis `loss_1.backward()` accumule par-dessus.
   Vérifié par la parité d'orchestration inline (Tier 4-light, 1e-5).
3. **2nd-order gelé pendant Grammar-A** (`training()` force `meta=False`, source L969 — que
   l'étudiant lui-même flaggait « seems inconsistent with parameter »). Reproduit sciemment ;
   vérifié (le 2nd-order ne bouge pas en training).
4. **Décodeur `make_chunked_sigmoid(6)`** (winner-takes-all par lettre), vs `global_sigmoid`
   de Blindsight.
5. **RangerVA** (paper Table 10) + fallback ADAMAX.
6. Déterminisme : même seed → courbes de perte identiques (garantie de reproductibilité).

### Décision restée en suspens (à trancher, cf. D13.2)
`config/domains/agl/training.yaml` a `second_order.hidden_dim: 48` — c'est le choix
**« production » du Sprint-08** qui *restaure* la couche cachée Pasquali. Le code source
(fidèle) l'a **absente** (même bug que Blindsight D.25). Le trainer étant piloté par config,
les deux marchent ; la version fidèle-source (`hidden_dim: null`) reste à acter au moment où
on voudra une parité *numérique* stricte vs `agl_tmlr.py`. Non bloquant pour Sprint 13
(mécaniques + reproductibilité validées).

### Hors scope (reporté)
- Reproduction empirique des chiffres Tables 5b/5c (20 réseaux × 12/3 epochs × seeds) — à
  lancer post-Sprint 13 (le disque projet plein bloque les runs, pas les tests).
- Parité *numérique* stricte bout-en-bout vs `agl_tmlr.py` (dépend de la décision hidden_dim).
- SARL → Sprint 14 (branche `refactor/sarl` depuis `6f07916`).
