"""AGL trainer — paper §4 + §A.2, Tables 5b/5c, Table 10.

:class:`AGLTrainer` orchestrates one ``(seed, setting)`` run of the 3-phase AGL
protocol:

1. **pre-train** on random-grammar words — the first-order network learns
   reconstruction, the second-order network learns to wager on first-order
   success (:meth:`AGLTrainer.pre_train`);
2. **reset** the first-order network to its *initial* weights (the mechanism of
   the High/Low awareness dissociation — ``D-agl-reset``, source L751);
3. **train** on Grammar-A words for N epochs (:meth:`AGLTrainer.training`, via
   the shared :func:`_run_training_loop`), keeping the second-order network
   **frozen** (source L969 hardcodes ``meta = False`` inside ``training()``).

The 20-cell High/Low replication (paper §A.2) lives in
:mod:`maps.domains.agl.pool`, which reuses :func:`_run_training_loop`.

D13.3 — refactored class instead of the student's inline notebook functions.
Parity bit-exact is guaranteed by tests (Sprint 13.G), not by 1-to-1 LOC.

**Load-bearing two-loss gradient pattern** (source ``pre_train`` L668-733,
identical to Blindsight): ``optimizer_1.zero_grad()`` **before**
``loss_2.backward(retain_graph=True)``; ``optimizer_2.step()`` then
``optimizer_2.zero_grad()``; finally ``loss_1.backward()`` accumulates *on top
of* the first-order gradients already deposited by ``loss_2`` — so
``optimizer_1.step()`` uses the sum. This cross-task coupling is not in the
paper text; it is read from the student code and preserved.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §4, Table 5b/5c, Table 10.
Dienes, Z. (1997). Transfer of implicit knowledge across domains.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR

from maps.core.losses import cae_loss, wagering_bce_loss
from maps.core.second_order import SecondOrderNetwork
from maps.domains.agl.data import array_words, target_second
from maps.networks.first_order_mlp import FirstOrderMLP, make_chunked_sigmoid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings registry — 6-cell factorial (same schema as BlindsightSetting)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AGLSetting:
    """One row of the paper Table 5b/5c factorial.

    ``cascade_1st`` / ``cascade_2nd`` map to the student ``type_cascade``
    dispatch (1 = both, 2 = 1st only, 3 = 2nd only, else none).
    """

    id: str
    label: str
    cascade_1st: bool
    cascade_2nd: bool
    second_order: bool


SETTINGS_REGISTRY: dict[str, AGLSetting] = {
    "setting-1-baseline": AGLSetting(
        "setting-1-baseline", "baseline (no cascade, no 2nd-order)", False, False, False
    ),
    "setting-2-cascade-1st": AGLSetting(
        "setting-2-cascade-1st", "cascade 1st only", True, False, False
    ),
    "setting-3-second-order-only": AGLSetting(
        "setting-3-second-order-only", "2nd-order only (no cascade)", False, False, True
    ),
    "setting-4-maps-1st": AGLSetting(
        "setting-4-maps-1st", "MAPS (cascade 1st + 2nd-order)", True, False, True
    ),
    "setting-5-cascade-2nd": AGLSetting(
        "setting-5-cascade-2nd", "cascade 2nd only", False, True, True
    ),
    "setting-6-full-maps": AGLSetting(
        "setting-6-full-maps", "Full MAPS (cascade both + 2nd-order)", True, True, True
    ),
}


# ---------------------------------------------------------------------------
# Optimizer factory (RangerVA preferred per paper Table 10; ADAMAX fallback)
# ---------------------------------------------------------------------------


_OPTIMIZER_REGISTRY: dict[str, type] = {
    "ADAM": optim.Adam,
    "ADAMAX": optim.Adamax,
    "ADAMW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSPROP": optim.RMSprop,
    "ADAGRAD": optim.Adagrad,
}

_RANGER_NAMES = {"RANGER", "RANGERVA", "RANGERQH", "RADAM"}

try:  # torch-optimizer ships RangerVA (paper Table 10 optimizer)
    import torch_optimizer as _optim2

    _OPTIMIZER_REGISTRY["RANGER"] = _optim2.Ranger
    _OPTIMIZER_REGISTRY["RANGERVA"] = _optim2.RangerVA
    _OPTIMIZER_REGISTRY["RANGERQH"] = _optim2.RangerQH
    _OPTIMIZER_REGISTRY["RADAM"] = _optim2.RAdam
    _HAS_TORCH_OPTIMIZER = True
except ImportError:  # pragma: no cover — exercised only without the `agl` extra
    _HAS_TORCH_OPTIMIZER = False


def _build_optimizer(name: str, params, lr: float) -> optim.Optimizer:
    """Construct an optimizer by string name (mirrors student dispatch).

    Falls back to ADAMAX with a warning when a Ranger variant is requested but
    ``torch-optimizer`` is not installed (``uv sync --extra agl``).
    """
    key = name.upper()
    if key not in _OPTIMIZER_REGISTRY:
        if key in _RANGER_NAMES and not _HAS_TORCH_OPTIMIZER:
            logger.warning(
                "optimizer %r requires torch-optimizer (uv sync --extra agl); "
                "falling back to ADAMAX",
                name,
            )
            return optim.Adamax(params, lr=lr)
        raise ValueError(
            f"unknown optimizer {name!r}; expected one of {sorted(_OPTIMIZER_REGISTRY)}"
        )
    return _OPTIMIZER_REGISTRY[key](params, lr=lr)


# ---------------------------------------------------------------------------
# Precision metric — verbatim calculate_metrics (agl_tmlr.py:451-489)
# ---------------------------------------------------------------------------


def calculate_precision(
    patterns_tensor: Tensor, output_first_order: Tensor, *, bits_per_letter: int = 6
) -> float:
    """Winner-takes-all reconstruction precision (TP / (TP + FP)).

    Per 6-bit chunk, the highest decoder activation (if > 0.1) is set to 1; the
    predicted one-hot pattern is compared to the input. Verbatim
    ``calculate_metrics`` — returns ``precision`` only.
    """
    output_first_order = output_first_order.detach()
    predicted_patterns = []
    for pattern in output_first_order:
        predicted_pattern = torch.zeros_like(pattern)
        for i in range(0, len(pattern), bits_per_letter):
            pack = pattern[i : i + bits_per_letter]
            max_index = torch.argmax(pack)
            if pattern[i + max_index] > 0.1:
                predicted_pattern[i + max_index] = 1
        predicted_patterns.append(predicted_pattern)
    predicted_patterns_tensor = torch.stack(predicted_patterns)

    tp = fp = 0.0
    for i in range(len(patterns_tensor)):
        tp += (patterns_tensor[i] * predicted_patterns_tensor[i]).sum().item()
        fp += ((1 - patterns_tensor[i]) * predicted_patterns_tensor[i]).sum().item()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


# ---------------------------------------------------------------------------
# Network cell + shared Grammar-A training loop
# ---------------------------------------------------------------------------


@dataclass
class NetworkCell:
    """One independent (1st-order, 2nd-order) pair + its optimizers/schedulers.

    Used both by :meth:`AGLTrainer.training` (a single cell) and by
    :class:`maps.domains.agl.pool.AGLNetworkPool` (N cells).
    """

    first_order: FirstOrderMLP
    second_order: SecondOrderNetwork
    optimizer_1: optim.Optimizer
    optimizer_2: optim.Optimizer
    scheduler_1: StepLR
    scheduler_2: StepLR


def _run_training_loop(
    cells: list[NetworkCell],
    n_epochs: int,
    *,
    rate_1: float,
    iters_1: int,
    rate_2: float,
    iters_2: int,
    batch_size: int,
    cae_lambda: float,
    bits_per_letter: int = 6,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grammar-A training loop shared by the single-cell trainer and the pool.

    **The second-order network is frozen** — source ``training()`` L969
    hardcodes ``meta = False``, overriding its own parameter (the student even
    flagged it "seems inconsistent with parameter"). We reproduce this
    faithfully (D13.2, ``train_meta_frozen_in_training``): the 2nd-order net
    forwards (so ``loss_2`` is recorded for metrics and its dropout consumes
    RNG identically) but is never back-propagated or stepped.

    Returns ``(losses_1, losses_2, precision)``, each shaped
    ``(len(cells), n_epochs)``.
    """
    n = len(cells)
    losses_1 = np.zeros((n, n_epochs))
    losses_2 = np.zeros((n, n_epochs))
    precision = np.zeros((n, n_epochs))

    for idx, cell in enumerate(cells):
        for epoch in range(n_epochs):
            patterns = array_words(2, batch_size, device=device)  # Grammar A

            h1: Tensor | None = None
            h2: Tensor | None = None
            for _ in range(iters_1):
                h1, h2 = cell.first_order(patterns, h1, h2, rate_1)
            assert h1 is not None and h2 is not None

            cell.optimizer_1.zero_grad()

            # 2nd-order forward for metrics only (frozen — meta=False, source L969).
            comparison: Tensor | None = None
            wager: Tensor | None = None
            for _ in range(iters_2):
                wager, comparison = cell.second_order(patterns, h2, comparison, rate_2)
            assert wager is not None
            target = target_second(patterns, h2, device=device)
            loss_2 = wagering_bce_loss(wager.squeeze(), target, reduction="sum")

            loss_1 = cae_loss(
                weight=cell.first_order.fc1.weight,
                x=patterns,
                recons_x=h2,
                hidden=h1,
                lam=cae_lambda,
                recon="bce_sum",
            )
            loss_1.backward()
            cell.optimizer_1.step()
            cell.scheduler_1.step()

            losses_1[idx, epoch] = loss_1.item()
            losses_2[idx, epoch] = loss_2.item()
            precision[idx, epoch] = calculate_precision(
                patterns, h2, bits_per_letter=bits_per_letter
            )

    return losses_1, losses_2, precision


# ---------------------------------------------------------------------------
# AGLTrainer
# ---------------------------------------------------------------------------


class AGLTrainer:
    """One ``(seed, setting)`` AGL run: build → pre_train (+reset) → training."""

    def __init__(
        self,
        setting: AGLSetting,
        seed: int,
        cfg: DictConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.setting = setting
        self.seed = seed
        self.cfg = cfg
        self.device = torch.device(device) if isinstance(device, str) else device

        self.first_order: FirstOrderMLP | None = None
        self.second_order: SecondOrderNetwork | None = None
        self.optimizer_1: optim.Optimizer | None = None
        self.optimizer_2: optim.Optimizer | None = None
        self.scheduler_1: StepLR | None = None
        self.scheduler_2: StepLR | None = None
        # Cached in build() — the initial 1st-order weights restored after pre-train.
        self._initial_first_order_state: dict | None = None

    # ---- cascade dispatch ----------------------------------------------

    def _cascade_params(self) -> tuple[float, int, float, int]:
        """Resolve (rate_1, iters_1, rate_2, iters_2) from the setting flags."""
        alpha = self.cfg.cascade.alpha
        n_iter = self.cfg.cascade.n_iterations
        rate_1 = alpha if self.setting.cascade_1st else 1.0
        iters_1 = n_iter if self.setting.cascade_1st else 1
        rate_2 = alpha if self.setting.cascade_2nd else 1.0
        iters_2 = n_iter if self.setting.cascade_2nd else 1
        return rate_1, iters_1, rate_2, iters_2

    def _bits_per_letter(self) -> int:
        return int(self.cfg.get("bits_per_letter", 6))

    # ---- build ----------------------------------------------------------

    def _make_first_order(self) -> FirstOrderMLP:
        return FirstOrderMLP(
            input_dim=self.cfg.first_order.input_dim,
            hidden_dim=self.cfg.first_order.hidden_dim,
            decoder_activation=make_chunked_sigmoid(self._bits_per_letter()),
            encoder_dropout=self.cfg.first_order.encoder_dropout,
            weight_init_range=tuple(self.cfg.first_order.weight_init_range),
        ).to(self.device)

    def _make_second_order(self) -> SecondOrderNetwork:
        hidden = self.cfg.second_order.get("hidden_dim", None) or None
        return SecondOrderNetwork(
            input_dim=self.cfg.second_order.input_dim,
            n_wager_units=self.cfg.second_order.n_wager_units,
            hidden_dim=hidden,
            dropout=self.cfg.second_order.dropout,
        ).to(self.device)

    def build(self) -> None:
        """Instantiate networks, pre-train optimizers/schedulers, cache init weights."""
        self.first_order = self._make_first_order()
        self.second_order = self._make_second_order()

        self.optimizer_1 = _build_optimizer(
            self.cfg.optimizer.name,
            self.first_order.parameters(),
            self.cfg.optimizer.lr_first_order,
        )
        self.optimizer_2 = _build_optimizer(
            self.cfg.optimizer.name,
            self.second_order.parameters(),
            self.cfg.optimizer.lr_second_order,
        )
        if self.cfg.scheduler.name != "StepLR":
            raise NotImplementedError(f"only StepLR supported; got {self.cfg.scheduler.name!r}")
        self.scheduler_1 = StepLR(
            self.optimizer_1, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma
        )
        self.scheduler_2 = StepLR(
            self.optimizer_2, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma
        )

        # D-agl-reset: snapshot the *initial* 1st-order weights (source L608).
        self._initial_first_order_state = copy.deepcopy(self.first_order.state_dict())

        logger.info(
            "Built AGLTrainer setting=%s seed=%d device=%s", self.setting.id, self.seed, self.device
        )

    def _ensure_built(self) -> None:
        if self.first_order is None or self.second_order is None:
            raise RuntimeError("call .build() before .pre_train() / .training()")

    # ---- pre-train (+ reset) -------------------------------------------

    def pre_train(self, n_epochs: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random-grammar pre-training, then reset the 1st-order to initial weights.

        Returns ``(losses_1, losses_2, precision)`` per epoch.
        """
        self._ensure_built()
        assert self.first_order is not None and self.second_order is not None
        assert self.optimizer_1 is not None and self.optimizer_2 is not None
        assert self.scheduler_1 is not None and self.scheduler_2 is not None
        assert self._initial_first_order_state is not None

        n_epochs = n_epochs if n_epochs is not None else self.cfg.train.n_epochs_pretrain
        batch_size = self.cfg.train.batch_size_pretrain
        cae_lambda = self.cfg.losses.cae_lambda
        bpl = self._bits_per_letter()
        rate_1, iters_1, rate_2, iters_2 = self._cascade_params()

        losses_1 = np.zeros(n_epochs)
        losses_2 = np.zeros(n_epochs)
        precision = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            patterns = array_words(1, batch_size, device=self.device)  # random grammar

            h1: Tensor | None = None
            h2: Tensor | None = None
            for _ in range(iters_1):
                h1, h2 = self.first_order(patterns, h1, h2, rate_1)
            assert h1 is not None and h2 is not None

            # --- two-loss gradient pattern (source L668-733) ---
            self.optimizer_1.zero_grad()

            if self.setting.second_order:
                comparison: Tensor | None = None
                wager: Tensor | None = None
                for _ in range(iters_2):
                    wager, comparison = self.second_order(patterns, h2, comparison, rate_2)
                assert wager is not None
                target = target_second(patterns, h2, device=self.device)
                loss_2 = wagering_bce_loss(wager.squeeze(), target, reduction="sum")
                loss_2.backward(retain_graph=True)
                self.optimizer_2.step()
                self.scheduler_2.step()
                self.optimizer_2.zero_grad()
                losses_2[epoch] = loss_2.item()
            else:
                # No-grad forward — preserves RNG (dropout) consumption for parity.
                with torch.no_grad():
                    comparison_ng: Tensor | None = None
                    for _ in range(iters_2):
                        _, comparison_ng = self.second_order(patterns, h2, comparison_ng, rate_2)

            loss_1 = cae_loss(
                weight=self.first_order.fc1.weight,
                x=patterns,
                recons_x=h2,
                hidden=h1,
                lam=cae_lambda,
                recon="bce_sum",
            )
            loss_1.backward()
            self.optimizer_1.step()
            self.scheduler_1.step()

            losses_1[epoch] = loss_1.item()
            precision[epoch] = calculate_precision(patterns, h2, bits_per_letter=bpl)

        # D-agl-reset (source L751): reload the initial 1st-order weights so the
        # Grammar-A training phase re-learns from scratch — the High/Low
        # awareness dissociation mechanism.
        self.first_order.load_state_dict(self._initial_first_order_state)

        return losses_1, losses_2, precision

    # ---- Grammar-A training (single cell) ------------------------------

    def training(self, n_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train the (reset) 1st-order on Grammar-A for ``n_epochs`` (single cell).

        Builds fresh *training-phase* optimizers/schedulers (source
        ``create_networks`` uses ``lr_training_*``) for this trainer's own
        networks and runs the shared :func:`_run_training_loop` on one cell.
        Returns per-epoch ``(losses_1, losses_2, precision)`` (1-D, squeezed
        from the single-cell arrays).
        """
        self._ensure_built()
        assert self.first_order is not None and self.second_order is not None

        opt1 = _build_optimizer(
            self.cfg.optimizer.name,
            self.first_order.parameters(),
            self.cfg.optimizer.lr_training_first_order,
        )
        opt2 = _build_optimizer(
            self.cfg.optimizer.name,
            self.second_order.parameters(),
            self.cfg.optimizer.lr_training_second_order,
        )
        sched1 = StepLR(
            opt1, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma
        )
        sched2 = StepLR(
            opt2, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma
        )
        cell = NetworkCell(self.first_order, self.second_order, opt1, opt2, sched1, sched2)

        rate_1, iters_1, rate_2, iters_2 = self._cascade_params()
        losses_1, losses_2, precision = _run_training_loop(
            [cell],
            n_epochs,
            rate_1=rate_1,
            iters_1=iters_1,
            rate_2=rate_2,
            iters_2=iters_2,
            batch_size=self.cfg.train.batch_size_training,
            cae_lambda=self.cfg.losses.cae_lambda,
            bits_per_letter=self._bits_per_letter(),
            device=self.device,
        )
        return losses_1[0], losses_2[0], precision[0]
