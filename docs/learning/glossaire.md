# Glossaire MAPS

> Vocabulaire technique du projet — concepts du paper, architectures,
> hyperparams, jargon du port, termes computationnels.
>
> Convention : **terme** : définition courte → *(contexte / réf paper /
> analogie neuro si pertinente)*.
>
> S'enrichit au fil des sprints — section "Ajouts du Sprint X" en fin
> de chaque catégorie au besoin.
>
> Inspiré du pattern Satori `docs/glossaire.md` (avec analogies neuro
> entre parenthèses pour les termes ML/RL).

---

## 1. Concepts MAPS centraux (paper §2.1, §2.2)

- **Cascade** ou **cascade model** : dynamique d'accumulation graduelle d'évidence
  selon `a(t) = α·new + (1-α)·a(t-1)`. Origine McClelland (1989). *(≈ un
  "running average" pondéré, comme une intégration leaky-IF en neuro
  computational.)*
- **First-order network** : l'autoencoder qui traite la tâche perceptive
  primaire (input → reconstruction). Branche encoder + decoder + hidden h.
- **Second-order network** : réseau métacognitif qui *observe* le first-order
  via la comparator matrix et produit un wager (confiance). Pasquali &
  Cleeremans (2010).
- **Comparator matrix** (`C_t`) : résidu input − reconstruction
  (`X_t - Ŷ_t`). Eq.1 du paper. Sortie du first-order vers le second-order.
- **Wager** (`W_t`) : sortie scalaire (ou 2-unit) du wagering head — probabilité
  perçue que la décision first-order est correcte. *(≈ confidence rating en
  expé conscience ; Koch & Preuschoff 2007 "betting".)*
- **Wagering head** : couche linéaire (ou Linear→ReLU→Linear avec hidden
  Pasquali) qui produit le wager depuis le comparator post-cascade.
- **Cascade iterations** : nombre de fois où on déroule la cascade par
  forward step. Paper = 50. = `1/α` par convention.
- **Cascade rate** (`α`) : taux d'intégration de la cascade. Paper = 0.02.
  ⚠️ **À ne pas confondre** avec le `α=0.45` de l'EMA wager.
- **Pasquali hidden** : couche cachée optionnelle dans le wagering head
  (citée paper §2.2 "as in Pasquali & Cleeremans 2010" mais code student
  l'omettait — bug D.25/D.28 corrigé Sprint 11).
- **EMA wager** (`α=0.45`) : exponential moving average sur le wager
  (utilisée SARL pour targets). Eq.13 paper. Le 2ème `α` du projet.

## 2. Architectures (`src/maps/core/`, `networks/`, `domains/`)

- **`ComparatorMatrix`** : `nn.Module` stateless qui fait `X - Ŷ`. Eq.1.
  Sprint 11.
- **`WageringHead`** : readout linear (1 ou 2 units) avec option `hidden_dim`
  pour Pasquali. Sprint 11.
- **`SecondOrderNetwork`** : composition Comparator → Dropout → cascade →
  Wager, partagé Blindsight + AGL. Sprint 11.
- **`SarlSecondOrderNetwork`** : variante SARL avec tied-weight decoder,
  dropout 0.1 (au lieu de 0.5), 2-unit raw logits. Vit dans
  `domains/sarl/` (Sprint 14). DETTE-1.
- **`FirstOrderMLP`** : encoder + decoder MLP partagé Blindsight/AGL.
  Sprint 12.
- **`SarlQNetwork`** : Q-network DQN avec cascade interne. Sprint 14.

## 3. Losses (`src/maps/core/losses.py`)

- **CAE** (Contractive AutoEncoder, Rifai 2011) : `recon(x, x̂) + λ·||J_h(x)||²_F`.
  Le first-order loss du student code, malgré la prose paper qui décrit
  SimCLR (cf. D-002).
- **NT-Xent / SimCLR** (Chen et al. 2020) : contrastive loss avec paires
  positives. *Le* first-order loss prose-paper, mais non utilisé student.
  Porté Sprint 11 D11.7 pour comparaison empirique post-Sprint 12.
- **Wagering BCE** : binary cross-entropy sur le wager post-sigmoid
  (paper eq.5).
- **Weight regularization** : L2 anchor entre student et frozen teacher,
  EWC-style sans Fisher weighting (Kirkpatrick 2017). C'est le vrai
  "distillation" utilisé SARL+CL.
- **`distillation_loss`** (KL Hinton 2015) : ⚠️ **DELETED Sprint 11 (D11.5)**.
  0 callers en prod. Récupérable via `git show main:src/maps/components/losses.py`.
- **`h(1-h)` sur ReLU** : quirk universel du code Vargas — la formule
  fermée du Jacobien CAE est la dérivée sigmoid, mais l'encoder est ReLU.
  Mathématiquement faux, préservé byte-for-byte pour parity (D-001).

## 4. Hyperparams paper (verrouillés dans `config/maps.yaml`)

| Param | Valeur | Source |
|-------|--------|--------|
| `cascade_rate` (α) | 0.02 | Paper §2.1 eq.6 |
| `n_iterations` | 50 | = 1/α, convention paper |
| `λ_CAE` | 1e-4 | Paper §2.1 |
| `dropout BS/AGL` | 0.5 | Student code |
| `dropout SARL` | 0.1 | Student SARL |
| `hidden_dim Pasquali BS` | 100 | D.25 restored |
| `hidden_dim Pasquali AGL` | 48 | D.28 restored |
| `wagering_units` | 1 (student) ou 2 (paper) | D-001 |
| `weight init readout` | uniform(0, 0.1) | Student |
| `EMA wager α` | 0.45 | Paper Table 11 eq.13 |
| `RANDOM_SEED` | 42 | Convention lab |

## 5. Domaines

- **Blindsight** : détection perceptive sous noise. Input 100-dim binaire.
  Sprint 12.
- **AGL** (Artificial Grammar Learning) : implicit/explicit learning.
  Input 48 bits (8 lettres × 6 bits). Sprint 13.
- **SARL** (Single-Agent RL) : DQN sur MinAtar. Games : Space Invaders,
  Breakout, Seaquest, Asterix, Freeway. Sprint 14.
- **SARL+CL** (continual learning) : DQN + teacher network + L2 anchor.
  Sprint 15.
- **MARL** (Multi-Agent RL) : MeltingPot 2.0 MAPPO. Substrates : Commons
  Harvest (closed/partnership), Chemistry, Territory inside-out.
  Linux-only. Sprint 16.
- **METTA-AI** : extension exploratoire, env externe `external/METTA/`.
  Sprint 16+ ou hors-scope.

## 6. Settings factoriels (paper Tables 5/6/7)

Combinatoire 2×2×2 sur (cascade 1st-order × cascade 2nd-order × meta) → 8
théoriques, 6 utilisés par le paper (7 = ACB pour SARL).

| Setting | Cascade 1st | Cascade 2nd | Meta (2nd order) | Notes |
|:-------:|:-----------:|:-----------:|:----------------:|-------|
| 1 | off | off | off | baseline |
| 2 | **on** | off | off | ⚠️ no-op sur SARL (D-sarl-cascade-noop) |
| 3 | off | off | **on** | meta sans cascade |
| 4 | **on** | off | **on** | |
| 5 | off | **on** | **on** | |
| 6 | **on** | **on** | **on** | full MAPS |
| 7 | (ACB Actor-Critic Baseline) | — | — | SARL only |

## 7. Termes du port / projet

- **Reverse-prompt** : pour chaque module Python existant, un .md qui
  répond *"quel aurait été le prompt qu'on aurait donné à Claude pour
  produire ce code ?"*. Format : spec → contraintes scientifiques →
  contraintes ingénierie → deviations/dettes → questions ouvertes →
  notes rewrite. Vivent dans `docs/learning/reverse-prompts/<chemin>.md`.
  41 produits en Sprint 10.
- **Walkthrough** : narratif post-rewrite expliquant *pourquoi* un
  module est conçu comme ça. Vivent dans `docs/learning/walkthroughs/`.
- **Brique** (emprunt à bilevel-fishery) : décomposition fine d'un
  sprint en unités plus granulaires. Chaque brique a sa table de
  reverse-prompts + corrections d'audit. À envisager Sprint 12+.
- **Deviation** (`D-*`) : écart paper ↔ code documenté dans
  `docs/reproduction/deviations.md`. Naming : `D-<domain>-<slug>`.
  Ex. `D-002` (SimCLR vs CAE), `D-sarl-cascade-noop`.
- **DETTE-*** : dette technique tracée. Pas un bug — un compromis à
  résoudre plus tard. DETTE-1 (doublon SO), DETTE-2 (doublon CAE),
  DETTE-3 ✅ résolue Sprint 11 (distillation_loss delete).
- **D11.x** : décisions du Sprint 11 Day 1. Format `D<sprint>.<num>`.
  D11.1 = rename `alpha` → `cascade_rate`. D11.7 = implémenter SimCLR.
- **Tier 1/2/3 parity** : niveau de garantie d'un test parity :
  - **Tier 1** = forward seul, bit-exact (modulo float32 noise)
  - **Tier 2** = forward + backward, gradients bit-exact
  - **Tier 3** = forward + backward + optimizer step, weights post-update
    bit-exact
- **`external/paper_reference/`** : code Vargas verbatim (44k LOC),
  source de vérité numérique. C'est CE code qui a produit les chiffres
  Tables 5/6/7 du paper — pas la version Sprint 09 sur main.
- **Sprint 11.5 / 11.6** : "demi-sprints" entre les vrais sprints,
  pour cleanup ou patterns transverses (config réorg, glossaire, etc.).
- **Phase X.A / X.B / ...** : sous-étapes d'un sprint, chacune avec
  son commit dédié. Sprint 11 a eu 7 phases (A locked decisions, B
  scaffold, C-E modules, F parity, G closeout).

## 8. Termes computationnels

- **MC-dropout** (Monte-Carlo dropout, Gal & Ghahramani 2016) : utiliser
  dropout en *inférence* pour estimer l'incertitude bayésienne.
  **Mécanisme caché de la cascade dans MAPS** : cascade × dropout =
  moyennage de ~50 masks dropout = MC-dropout déguisé. Sans dropout
  (SARL Q-network), la cascade est no-op (D-sarl-cascade-noop).
- **Tied weights** : décodeur dont les poids sont la transposée de
  l'encodeur (`W^T · h`). Utilisé par SARL pour économiser des params.
  Bloque l'extraction du comparator → DETTE-1.
- **Bit-flip augmentation** : pour SimCLR sur stimulus binaires,
  créer la paire positive en flippant aléatoirement p% des bits. Choix
  de recherche à acter Sprint 12 (D11.7).
- **`retain_graph=True`** : flag PyTorch `backward()` qui garde le
  graphe de calcul pour permettre un backward ultérieur. Coûteux en
  mémoire. MAPS l'utilise pour backward à travers les 50 itérations
  cascade.
- **Bootstrap** (dans cascade) : premier step (`t=0`) où il n'y a pas
  encore de `prev_activation` — la fonction retourne `new` directement.
- **Steady state** d'une cascade : valeur asymptotique
  `a(∞) → E[new]` quand la cascade tourne longtemps. Pour `new`
  constant : `a(∞) = new` (no-op).

## 9. Acronymes ML / RL / EEG

- **ACB** : Actor-Critic Baseline (SARL Setting 7, Young & Tian 2019)
- **BCE** : Binary Cross-Entropy
- **BPTT** : Backpropagation Through Time
- **CAE** : Contractive AutoEncoder (Rifai 2011)
- **CL** : Continual Learning
- **CNN** : Convolutional Neural Network
- **DQN** : Deep Q-Network (Mnih et al. 2015)
- **EMA** : Exponential Moving Average
- **EWC** : Elastic Weight Consolidation (Kirkpatrick 2017)
- **HPC** : High Performance Computing (DRAC Canada : Narval, Tamia)
- **KL** : Kullback-Leibler (divergence)
- **MAPS** : Metacognitive Architecture for Perceptual and Social Learning
- **MARL** : Multi-Agent RL
- **MLP** : Multi-Layer Perceptron
- **MPS** : Metal Performance Shaders (backend GPU Apple Silicon)
- **MSE** : Mean Squared Error
- **NT-Xent** : Normalized Temperature-scaled cross-Entropy (loss SimCLR)
- **PPO** : Proximal Policy Optimization (Schulman et al. 2017)
- **ReLU** : Rectified Linear Unit
- **RL** : Reinforcement Learning
- **RNG** : Random Number Generator
- **SARL** : Single-Agent RL
- **SimCLR** : Simple framework for Contrastive Learning of visual Representations (Chen 2020)
- **SLURM** : Simple Linux Utility for Resource Management (HPC scheduler)
- **YAGNI** : You Aren't Gonna Need It

## 10. Termes neuro pour cadrer les analogies

(Pour le lab, ces termes sont natifs — listés ici pour rappeler les
ancrages utilisés dans les docstrings et walkthroughs.)

- **Métacognition** : connaissance qu'a un sujet de ses propres états
  cognitifs. Le second-order de MAPS est un *modèle computationnel*
  de ça.
- **Wager** (consciousness studies) : montant qu'un sujet est prêt à
  parier sur sa propre réponse. Opérationnalisation de la confiance.
  Koch & Preuschoff (2007), Persaud et al. (2007).
- **Blindsight** : phénomène où un patient avec lésion V1 peut détecter
  des stimuli sans rapport conscient. MAPS le simule par signal
  first-order présent + wager bas.
- **Pasquali & Cleeremans (2010)** : "Know thyself: metacognitive
  networks and measures of consciousness". *Cognition*, 117(2), 182-190.
  Origine de l'archi Comparator + Wager (le hidden layer du wager
  vient de ce papier).
- **McClelland (1989)** : "Parallel distributed processing implications
  for cognition and development". Origine de la cascade. Modèle de
  dynamique d'activation pour des réseaux PDP.
- **Gal & Ghahramani (2016)** : "Dropout as a Bayesian approximation".
  Le papier qui légitime l'usage de dropout en inférence comme
  estimateur d'incertitude. *Le clé pour comprendre pourquoi cascade
  + dropout = MAPS marche.*

---

## Ajouts du Sprint 11.6 (2026-05-25) — création du glossaire

Première version. Sources extraites de :
- `CLAUDE.md` (constantes canoniques + structure)
- `docs/learning/reverse-prompts/core/{cascade,second_order,losses}.md`
- `docs/learning/walkthroughs/cascade-from-paper-to-code.md`
- `docs/reproduction/deviations.md` (D-*, DETTE-*)
- `docs/sprints/sprint-11-core-rewrite.md` (D11.x)

Termes ajoutés au passage : MC-dropout, NT-Xent, bit-flip augmentation,
Pasquali hidden, D-sarl-cascade-noop, Tier 1/2/3 parity, Brique
(emprunt bilevel-fishery), Sprint 11.5/11.6 demi-sprints.
