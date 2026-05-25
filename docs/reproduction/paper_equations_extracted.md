# Paper equations 1-18 — verbatim extractions

**Source :** `pdf/MAPS_TMLR_Journal_Submission.pdf` — Vargas et al., TMLR submission (2025), pp. 6-9.

Règle : équations copiées verbatim en notation paper. Description des variables citée du papier
quand elle est donnée explicitement ; inférée depuis le contexte sinon (et alors clairement
marquée *"(inféré)"*). Pointeur vers le port `src/maps/` quand il existe. Flags vers
`deviations.md` quand une divergence code↔paper est déjà connue ou détectée ici.

Phase B du plan `docs/plans/plan-20260419-review-and-reproduce.md` remplit ce doc (sub-phase B.5).
Document source de vérité pour Phase C (review composants) et Phase D (alignment config).

---

## §2.1 — Know-Thyself architecture (Blindsight / AGL), p. 6

### Equation 1 — Comparison matrix

**Paper location :** p. 6, après "See equations 1 to 3 to understand the metacognitive signal."

$$
\mathbf{C}_t = \mathbf{X}_t - \hat{\mathbf{Y}}^{(1)}_t
$$

**Variables (inféré du contexte §2.1) :**
- `X_t` — input de l'autoencoder à l'instant `t`.
- `Ŷ^{(1)}_t` — sortie reconstruite par le first-order network (autoencoder output).
- `C_t` — la comparison matrix = résidu input / reconstruction. Entrée du second-order.

**Port :** `src/maps/components/second_order.ComparatorMatrix.forward()`. Pour SARL, la comparaison
est calculée inline dans `SarlQNetwork.forward()` : `comparison = flat_input - reconstruction`.

### Equation 2 — Dropout sur comparison

**Paper location :** p. 6, ligne après eq. 1.

$$
\mathbf{C}'_t = \mathrm{Dropout}(\mathbf{C}_t)
$$

**Variables :**
- `C_t` — comparison matrix (sortie de eq. 1).
- `C'_t` — version masquée par dropout.

**Port :** `SarlSecondOrderNetwork.forward()` ligne 144 : `comparison_out = self.dropout(comparison_matrix)`.
Rate `p=0.1` (hardcodé dans le code student, paper silent — à documenter).

**⚠️ Flag D-sarl-dropout-position** (à ouvrir Phase B.12) : le papier applique le dropout **avant** la
cascade (eq. 2 puis eq. 3 sur C'_t déjà intégré cascade). Notre port applique le dropout **à chaque
itération** de la boucle cascade → 50× par update pour setting 6 (dominante profil 31% wall).
Probablement **pas paper-faithful**.

### Equation 3 — Wagering linear head

**Paper location :** p. 6, ligne après eq. 2.

$$
\mathbf{W}_t = (\mathbf{W} \mathbf{C}'_t + \mathbf{b})
$$

**Variables :**
- `W` — matrice de poids du layer wagering, shape `(2, input_dim)` (paper §2.1 "wagering units = 2",
  Koch & Preuschoff 2007).
- `b` — bias.
- `C'_t` — comparison matrix post-dropout (eq. 2).
- `W_t` — raw logits 2-unit du wager (pas de sigmoid/softmax ici, appliqués dans L_BCE eq. 5).

**Port :** `SarlSecondOrderNetwork.forward()` ligne 146 : `wager = self.wager(comparison_out)`.
Weight init `init.uniform_(self.wager.weight, 0.0, 0.1)` (student code, paper silent).

### Equation 4 — Contrastive loss (main task)

**Paper location :** p. 6, section "We employ a contrastive loss (equation 4) for the main task".

$$
\mathcal{L}_{\mathrm{contrastive}} = \ell_{i,j}
= -\log \frac{\exp\!\big(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau\big)}
              {\sum_{k=1}^{2N} \mathbf{1}_{[k \ne i]} \, \exp\!\big(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau\big)}
$$

**Variables (paper verbatim) :**
- `z_i, z_j` — latent representations (hidden units `h`) for samples `i` and `j`.
- `sim(·,·)` — similarity function.
- `τ` — temperature parameter controlling the sharpness of the distribution.
- `N` — batch size ; `2N` represents positive and negative pairs.

**Reference :** Chen et al. (2020), *A Simple Framework for Contrastive Learning of Visual
Representations (SimCLR)*, arXiv:2002.05709.

**🚨 Divergence majeure D-002** (already open dans `deviations.md`) : notre port **n'utilise pas**
cette loss SimCLR. Il utilise un **Contractive AutoEncoder loss (CAE, Rifai et al. 2011)** via
`src/maps/components/losses.cae_loss` et `src/maps/experiments/sarl/losses.cae_loss`. Deux losses
mathématiquement différentes malgré la collision de nom "contrastive". Décision politique Phase
C.7-C.9.

### Equation 5 — BCE loss (wagering)

**Paper location :** p. 6, "For wagering, we used a binary cross-entropy loss (equation 5) to handle
class imbalance."

$$
\mathcal{L}_{\mathrm{BCE}} = -\big[\, y \cdot \log(\sigma(\mathrm{logits})) + (1-y) \cdot \log\big(1 - \sigma(\mathrm{logits})\big) \,\big]
$$

**Variables (paper verbatim) :**
- `y ∈ {0, 1}` — ground truth binary label.
- `σ(·)` — sigmoid activation function : `σ(x) = 1 / (1 + e^{-x})`.
- `logits` — output de eq. 3 (raw logits 2-unit du wager head).

**Port :** `trainer.sarl_update_step` ligne 194 : `loss_second = F.binary_cross_entropy_with_logits(wager, targets_wagering)`.

**⚠️ Flag D-sarl-bce-shape** (à ouvrir Phase B.12) : paper eq. 5 opère sur `y` scalaire. Notre
port passe des `wager` shape `(B, 2)` et `targets_wagering` shape `(B, 2)` à
`binary_cross_entropy_with_logits`, ce qui applique BCE indépendamment sur les 2 unités. Match
approximatif (si target est one-hot les 2 BCE donnent la même info que 1 BCE scalaire) mais pas
littéralement eq. 5. Probablement OK, à documenter.

### Equation 6 — Cascade update

**Paper location :** p. 6, "We empirically selected 50 cascade iterations for all test cases (α = 0.02)."

$$
a_{ir}(t) = \alpha \sum_j w_{ij} \, a_{js}(t) + (1 - \alpha) \, a_{ir}(t - 1)
$$

**Variables (paper verbatim) :**
- `y ∈ {0, 1}` — ground truth binary label (carried from eq. 5 context).
- `σ(·)` — sigmoid activation function.
- `α ∈ [0, 1]` — Cascade rate parameter.
- `a_{ij}(t)` — activation of neuron `i` in layer `j` at time step `t`.
- `w_{ij}` — weight connection from neuron `i` to neuron `j`.

**Paper constants :** `α = 0.02`, `N_cascade = 1/α = 50` iterations.

**Port :** `src/maps/components/cascade.cascade_update(new, prev, alpha)` = `α·new + (1-α)·prev`.
`α·Σ_j w_{ij}·a_{js}(t)` du papier = la fresh activation déjà calculée par un layer linear+nonlin
dans le port → match sémantique.

---

## §2.2 — SARL architecture (MinAtar DQN), p. 7-8

Paper dit textuellement avant eq. 7 : *"For SARL (see Figure 3), we employ a DQN framework.
We use convolutional layers which allow for reduced computational complexity, a Q-network,
and a replay buffer for the learning stability. We use cascade model in the main network as in
equations 7 to 11."*

Puis : **"For i = 0, 1, 2, …, N_cascade − 1 :"**

### Equation 7 — Convolutional flattened feature

**Paper location :** p. 7.

$$
\mathbf{X}_{\mathrm{flat}} = \mathrm{Flatten}\big(\mathrm{ReLU}(\mathrm{Conv2d}(\mathbf{X}_{\mathrm{input}}))\big)
$$

**Variables :**
- `X_input` — MinAtar state (paper §A.3 : 10×10 grid with multiple channels per game).
- `Conv2d` — kernel 3×3, stride 1, 16 output channels (paper silent, confirmed via
  `external/paper_reference/sarl_maps.py` line 132).
- `X_flat` — flattened conv features, shape `(batch, 1024)` = 8×8 (conv output spatial) × 16 (channels).

**Port :** `SarlQNetwork.forward()` ligne 83-84 : `conv_out = F.relu(self.conv(x)); flat_input = conv_out.view(conv_out.size(0), -1)`.

### Equation 8 — Raw hidden activations

**Paper location :** p. 7.

$$
\mathbf{H}^{(i)}_{\mathrm{raw}} = \mathrm{ReLU}\big(\mathbf{W}_{\mathrm{hidden}} \, \mathbf{X}_{\mathrm{flat}} + \mathbf{b}_{\mathrm{hidden}}\big)
$$

**Variables :**
- `W_hidden, b_hidden` — weight + bias du fully-connected hidden layer (1024 → 128).
- `H^{(i)}_raw` — fresh hidden activations à l'itération `i` de la boucle cascade (128-dim).

**Port :** `SarlQNetwork.forward()` ligne 85 : `hidden = F.relu(self.fc_hidden(flat_input))`.

### Equation 9 — Cascade-integrated hidden

**Paper location :** p. 7.

$$
\mathbf{H}^{(i)} = \begin{cases}
  \alpha \cdot \mathbf{H}^{(i)}_{\mathrm{raw}} + (1 - \alpha) \cdot \mathbf{H}^{(i-1)} & \text{if}\ \mathbf{H}^{(i-1)} \ne \mathrm{None} \\
  \mathbf{H}^{(i)}_{\mathrm{raw}} & \text{otherwise}
\end{cases}
$$

**Variables :**
- `α` — cascade rate (= 0.02).
- `H^{(i-1)}` — cascade state de l'itération précédente (None à i=0, sinon propagé par le caller).
- `H^{(i)}` — cascade-integrated hidden après itération `i`.

**Port :** `cascade.cascade_update(new=hidden, prev=prev_h2, alpha=cascade_rate)` appelé depuis
`SarlQNetwork.forward()` ligne 86. Le `if H^{(i-1)} ≠ None` est dans `cascade_update` même
(ligne 64) : `if prev_activation is None: return new_activation`.

### Equation 10 — Q-values output

**Paper location :** p. 7.

$$
\mathbf{Q}^{(i)} = \mathbf{W}_{\mathrm{actions}} \, \mathbf{H}^{(i)} + \mathbf{b}_{\mathrm{actions}}
$$

**Variables :**
- `W_actions, b_actions` — weight + bias du DQN output head (128 → num_actions).
- `Q^{(i)}` — Q-values vector à l'itération `i`.

**Port :** `SarlQNetwork.forward()` ligne 88 : `q_values = self.actions(hidden)`.

### Equation 11 — Placeholder (empty)

**Paper location :** p. 7, entre eq. 10 et eq. 12.

**Statut :** le papier a le label `(11)` sur une ligne, **sans contenu mathématique**. Vraisemblable
artefact LaTeX (numérotation d'une équation absente, ou ligne blanche dans un aligned environment).

**Décision :** pas d'équation à reporter. Cette sous-section existe uniquement pour maintenir le
nombre "18 équations" du plan. À mentionner dans `deviations.md` comme "paper typesetting artifact,
no semantic content" pour référence future.

### Equation 12 — Tied-weight reconstruction

**Paper location :** p. 7, *"We then compute the comparison matrix using the inputs and a reconstructed
output using H^{(i)} and the transpose of the weights at H^{(i)}_raw (tied weights) as to maintain
the number of parameters, and reduce complexity (see equation 12)."*

$$
\hat{\mathbf{X}}_{\mathrm{recon}} = \mathrm{ReLU}\big(\mathbf{W}^{\top}_{\mathbf{H}^{(i)}_{\mathrm{raw}}} \, \mathbf{H}^{(i)} + \mathbf{b}_{\mathrm{recon}}\big)
$$

**Variables :**
- `W_{H^{(i)}_raw}` — la matrice de weights du layer hidden (eq. 8). Sa transposée sert d'encoder→decoder.
- `b_recon` — bias additif de la reconstruction.
- `H^{(i)}` — hidden cascade-integrated.

**Port :** `SarlQNetwork.forward()` ligne 91 : `reconstruction = F.relu(F.linear(hidden, self.fc_hidden.weight.t()))`.

**⚠️ Flag D-sarl-recon-bias** (à ouvrir Phase B.12) : paper écrit `+ b_recon` explicitement. Notre
port **omet le bias**. `F.linear(hidden, weight.t())` sans `bias=` argument → pas de bias appliqué.
Le student `sarl_maps.py` ligne 180 idem. **Divergence paper ↔ student ↔ port commune, à trancher
Phase D.3.**

### Equation 13 — Wagering EMA

**Paper location :** p. 8, *"α was found empirically and was used for both SARL, SARL + CL, and
MARL. For MARL, the wagering signal was calculated independently for every agent."*

$$
\mathrm{EMA}_t = \alpha \cdot r_t + (1 - \alpha) \cdot \mathrm{EMA}_{t-1}
$$

**Variables :**
- `r_t` — reward au step `t` (par agent pour MARL).
- `α` — smoothing factor **= 0.45** (Table 11 p. 30). **Distinct** du `α = 0.02` cascade (eq. 6) —
  le papier réutilise le symbole pour deux notions.

**Port :** `src/maps/experiments/sarl/data.target_wager()` lignes 107-113.

**🚨 Divergence majeure D-sarl-alpha-ema** (à ouvrir Phase B.12) : notre `target_wager` fait
`scaled_alpha = float(alpha / 100)` puis la récurrence avec `scaled_alpha`. Config passe
`alpha=1.0` → effectif `α = 0.01` (45× smaller than paper 0.45). Le student shell `-ema 25` donnerait
`α = 0.25` (encore différent). **Triple divergence paper / student / port.**

### Equation 14 — Wagering label

**Paper location :** p. 8.

$$
y^{(i)}_{\mathrm{wager}}(t) = \begin{cases}
  (1, 0) & \text{if}\ r^{(i)}_t > \mathrm{EMA}^{(i)}_t \quad \text{(high wager)} \\
  (0, 1) & \text{if}\ r^{(i)}_t \le \mathrm{EMA}^{(i)}_t \quad \text{(low wager)}
\end{cases}
$$

**Variables :**
- `r^{(i)}_t` — reward du sample `i` à l'instant `t`.
- `EMA^{(i)}_t` — EMA du sample `i` à l'instant `t` (eq. 13).
- `y^{(i)}_wager(t)` — one-hot label 2-dim pour la cible BCE (eq. 5).

**Port :** `data.target_wager()` lignes 114-117 : `if g > ema: new_tensor[i] = [1, 0] else [0, 1]`.
Match exact (incluant le tie-break strict `>` : égalité `r_t == EMA_t` → low wager).

---

## §2.3 — Continual Learning, p. 9

Paper dit avant eq. 15 : *"The weight regularization loss (Equation 15) constrains parameter drift
by penalizing deviations of current network parameters θ_k from their teacher network counterparts.
The feature preservation loss (Equation 16) maintains representational similarity between the
hidden states h^{student}_1 of the current network and the teacher network. The task-specific loss
(Equation 17) varies depending on whether training the main network (contrastive loss) or
second-order network (binary cross-entropy loss). These components are combined in our continual
learning objective (Equation 18) with normalized weights that sum to one, ensuring balanced
contribution from each loss component during the learning process."*

### Equation 15 — Weight regularization loss

**Paper location :** p. 9.

$$
\mathcal{L}_{\mathrm{reg}} = \frac{1}{\max_t(\mathcal{L}_{\mathrm{reg}}(t))} \sum_k \| \theta_k - \theta^{\mathrm{teacher}}_k \|_2^2
$$

**Variables (paper verbatim) :**
- `θ_k` — parameters of the current network (layer `k`).
- `θ^{teacher}_k` — teacher network parameters (frozen, from previous task).
- `max_t(L_reg(t))` — running max over epochs (normalisation dynamique).

**Port :** `src/maps/experiments/sarl_cl/trainer.compute_weight_regularization()`. Normalisation
via `src/maps/experiments/sarl_cl/loss_weighting.DynamicLossWeighter`.

### Equation 16 — Feature preservation loss

**Paper location :** p. 9.

$$
\mathcal{L}_{\mathrm{feature}} = \frac{1}{\max_t(\mathcal{L}_{\mathrm{feature}}(t))} \cdot \frac{1}{n} \sum_{i=1}^{n} \big( \mathbf{h}^{\mathrm{student}}_1(i) - \mathbf{h}^{\mathrm{teacher}}_1(i) \big)^2
$$

**Variables (paper verbatim) :**
- `h^{network}_1` — hidden states of the current network (layer 1).
- `h^{student}_1(i), h^{teacher}_1(i)` — sample `i`, student vs teacher.
- `n` — batch size.

**Port :** `sarl_cl/trainer.py` + `loss_weighting.py`.

### Equation 17 — Task loss (multi-use)

**Paper location :** p. 9.

$$
\mathcal{L}_{\mathrm{task}} = \frac{1}{\max_t(\mathcal{L}_{\mathrm{task}}(t))} \cdot \begin{cases}
  \mathcal{L}_{\mathrm{contrastive}} & \text{for main network} \\
  \mathcal{L}_{\mathrm{BCE}} & \text{for second-order network}
\end{cases}
$$

**Variables :**
- `L_contrastive` — eq. 4 (SimCLR-style — voir D-002).
- `L_BCE` — eq. 5.
- Normalisation running max idem eq. 15, 16.

**Port :** `sarl_cl/trainer.py` (dispatch main vs second-order).

### Equation 18 — Continual learning total loss

**Paper location :** p. 9.

$$
\mathcal{L}_{\mathrm{continual}} = \lambda_{\mathrm{task}} \, \mathcal{L}_{\mathrm{task}} + \lambda_{\mathrm{reg}} \, \mathcal{L}_{\mathrm{reg}} + \lambda_{\mathrm{feature}} \, \mathcal{L}_{\mathrm{feature}}
$$

**Variables (paper verbatim) :**
- `λ_task + λ_reg + λ_feature = 1` (normalized weights that sum to one).

**Paper Table 11 values :** `λ_task = 0.3`, `λ_reg = 0.6`, `λ_feature = 0.1`.

**⚠️ Incohérence paper** : texte p. 17 (CL Results) mentionne un "optimal" **différent** —
`task = 0.4, reg = 0.4, feature = 0.2`. Table et texte disagree. À flagger dans `deviations.md`
comme D-cl-weights-inconsistency.

**🚨 Divergence port D-cl-weights** : notre `config/training/sarl_cl.yaml` = `(1.0, 1.0, 1.0)`
(unnormalized). Ni aligné sur Table 11, ni sur texte p. 17. À fixer Phase D.20.

---
