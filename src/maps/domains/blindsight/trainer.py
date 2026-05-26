"""Blindsight trainer — paper §3, Table 5a.

:class:`BlindsightTrainer` orchestrates one ``(seed, setting)``
combination : build networks + optimizers + schedulers, pre-train,
evaluate per condition.

D12.5 — refactored class (``build/train/evaluate``) instead of student
verbatim 522-line inline. Parity bit-exact maintained by tests
(Sprint 12.G), not by 1-to-1 LOC.

**Load-bearing two-loss gradient pattern** (paper §3 + student
``blindsight_tmlr.py:pre_train``) :

1. ``optimizer_1.zero_grad()`` **before** any backward
2. If ``meta`` : forward 2nd-order, ``loss_2.backward(retain_graph=True)``,
   ``optimizer_2.step()``, ``scheduler_2.step()``. This gradient flow
   accumulates into the 1st-order network's parameters too.
3. CAE loss on 1st-order : ``loss_1.backward(retain_graph=True)`` —
   adds to the already-accumulated grads from step 2.
4. ``optimizer_1.step()`` uses the **sum** of both backward passes.
5. ``scheduler_1.step()``.

Skip step 2 (when ``second_order=False``) requires a no-grad forward
to preserve the RNG consumption pattern (parity with student).

**6 settings (paper Table 5a)** — see :data:`SETTINGS_REGISTRY`.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3, Table 5a, Table 9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import DictConfig
from torch import optim
from torch.optim.lr_scheduler import StepLR

from maps.core.losses import cae_loss, simclr_loss, wagering_bce_loss
from maps.core.second_order import SecondOrderNetwork
from maps.domains.blindsight.augmentations import bit_flip
from maps.domains.blindsight.data import (
    ConditionParams,
    generate_patterns,
)
from maps.networks.first_order_mlp import FirstOrderMLP, global_sigmoid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings registry — D12.6 (6-cell schema only, no backward-compat 2x2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlindsightSetting:
    """One row of the paper Table 5a factorial.

    Attributes
    ----------
    id : str
        Slug, e.g. ``"setting-6-full-maps"``.
    label : str
        Human-readable label.
    cascade_1st : bool
        Cascade applied to the 1st-order encoder/decoder.
    cascade_2nd : bool
        Cascade applied to the 2nd-order comparator path.
    second_order : bool
        Whether the second-order network is trained.
    """

    id: str
    label: str
    cascade_1st: bool
    cascade_2nd: bool
    second_order: bool


SETTINGS_REGISTRY: dict[str, BlindsightSetting] = {
    "setting-1-baseline": BlindsightSetting(
        "setting-1-baseline",
        "baseline (no cascade, no 2nd-order)",
        cascade_1st=False,
        cascade_2nd=False,
        second_order=False,
    ),
    "setting-2-cascade-1st": BlindsightSetting(
        "setting-2-cascade-1st",
        "cascade 1st only",
        cascade_1st=True,
        cascade_2nd=False,
        second_order=False,
    ),
    "setting-3-second-order-only": BlindsightSetting(
        "setting-3-second-order-only",
        "2nd-order only (no cascade)",
        cascade_1st=False,
        cascade_2nd=False,
        second_order=True,
    ),
    "setting-4-maps-1st": BlindsightSetting(
        "setting-4-maps-1st",
        "MAPS (cascade 1st + 2nd-order)",
        cascade_1st=True,
        cascade_2nd=False,
        second_order=True,
    ),
    "setting-5-cascade-2nd": BlindsightSetting(
        "setting-5-cascade-2nd",
        "cascade 2nd only",
        cascade_1st=False,
        cascade_2nd=True,
        second_order=True,
    ),
    "setting-6-full-maps": BlindsightSetting(
        "setting-6-full-maps",
        "Full MAPS (cascade both + 2nd-order)",
        cascade_1st=True,
        cascade_2nd=True,
        second_order=True,
    ),
}


# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    """Per-epoch losses tracked during :meth:`BlindsightTrainer.train`."""

    losses_1: np.ndarray  # CAE loss per epoch
    losses_2: np.ndarray  # BCE wager loss per epoch (zeros if no 2nd-order)


@dataclass
class EvalMetrics:
    """Per-condition metrics from :meth:`BlindsightTrainer.evaluate`."""

    discrimination_accuracy: dict[str, float] = field(default_factory=dict)
    wager_accuracy: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


_OPTIMIZER_REGISTRY = {
    "ADAM": optim.Adam,
    "ADAMAX": optim.Adamax,
    "ADAMW": optim.AdamW,
    "SGD": optim.SGD,
    "RMS": optim.RMSprop,
}


def _build_optimizer(name: str, params, lr: float):
    """Construct an optimizer by string name (matches student dispatch)."""
    name = name.upper()
    if name not in _OPTIMIZER_REGISTRY:
        raise ValueError(f"unknown optimizer {name!r}; expected one of {list(_OPTIMIZER_REGISTRY)}")
    return _OPTIMIZER_REGISTRY[name](params, lr=lr)


# ---------------------------------------------------------------------------
# BlindsightTrainer
# ---------------------------------------------------------------------------


class BlindsightTrainer:
    """One ``(seed, setting)`` Blindsight training run.

    Use the lifecycle :

    .. code-block:: python

        trainer = BlindsightTrainer(setting, seed, cfg, env_cfg)
        trainer.build()
        metrics = trainer.train()
        eval_metrics = trainer.evaluate()

    Parameters
    ----------
    setting : BlindsightSetting
        One of the 6 settings (lookup via :data:`SETTINGS_REGISTRY`).
    seed : int
        Random seed for this run.
    cfg : omegaconf.DictConfig
        **Merged & resolved** config — must contain both
        ``training.yaml`` (cascade/first_order/second_order/losses/
        optimizer/scheduler/train/first_order_loss) and ``env.yaml``
        (conditions/pre_training_condition/eval) sections. Caller
        (CLI) is responsible for the merge ; the trainer trusts the
        cfg is complete.
    device : torch.device, optional
        Default ``"cpu"``. Use :func:`maps.utils.device.get_device`.
    """

    def __init__(
        self,
        setting: BlindsightSetting,
        seed: int,
        cfg: DictConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.setting = setting
        self.seed = seed
        self.cfg = cfg
        self.device = torch.device(device) if isinstance(device, str) else device

        # Set after build()
        self.first_order: FirstOrderMLP | None = None
        self.second_order: SecondOrderNetwork | None = None
        self.optimizer_1: optim.Optimizer | None = None
        self.optimizer_2: optim.Optimizer | None = None
        self.scheduler_1: StepLR | None = None
        self.scheduler_2: StepLR | None = None

    # ---- cascade dispatch per setting -----------------------------------

    def _cascade_params(self) -> tuple[float, int, float, int]:
        """Resolve (rate_1, iters_1, rate_2, iters_2) from setting flags."""
        alpha = self.cfg.cascade.alpha
        n_iter = self.cfg.cascade.n_iterations  # = 1/alpha = 50

        rate_1 = alpha if self.setting.cascade_1st else 1.0
        iters_1 = n_iter if self.setting.cascade_1st else 1
        rate_2 = alpha if self.setting.cascade_2nd else 1.0
        iters_2 = n_iter if self.setting.cascade_2nd else 1
        return rate_1, iters_1, rate_2, iters_2

    # ---- build ----------------------------------------------------------

    def build(self) -> None:
        """Instantiate networks, optimizers, schedulers."""
        # Networks
        self.first_order = FirstOrderMLP(
            input_dim=self.cfg.first_order.input_dim,
            hidden_dim=self.cfg.first_order.hidden_dim,
            decoder_activation=global_sigmoid,
            encoder_dropout=self.cfg.first_order.encoder_dropout,
            weight_init_range=tuple(self.cfg.first_order.weight_init_range),
        ).to(self.device)

        self.second_order = SecondOrderNetwork(
            input_dim=self.cfg.second_order.input_dim,
            n_wager_units=self.cfg.second_order.n_wager_units,
            hidden_dim=self.cfg.second_order.hidden_dim or None,
            dropout=self.cfg.second_order.dropout,
        ).to(self.device)

        # Optimizers
        self.optimizer_1 = _build_optimizer(
            self.cfg.optimizer.name,
            self.first_order.parameters(),
            lr=self.cfg.optimizer.lr_first_order,
        )
        self.optimizer_2 = _build_optimizer(
            self.cfg.optimizer.name,
            self.second_order.parameters(),
            lr=self.cfg.optimizer.lr_second_order,
        )

        # Schedulers (StepLR only — paper Table 9)
        if self.cfg.scheduler.name != "StepLR":
            raise NotImplementedError(f"only StepLR supported; got {self.cfg.scheduler.name!r}")
        self.scheduler_1 = StepLR(
            self.optimizer_1,
            step_size=self.cfg.scheduler.step_size,
            gamma=self.cfg.scheduler.gamma,
        )
        self.scheduler_2 = StepLR(
            self.optimizer_2,
            step_size=self.cfg.scheduler.step_size,
            gamma=self.cfg.scheduler.gamma,
        )

        logger.info(
            "Built BlindsightTrainer setting=%s seed=%d device=%s",
            self.setting.id,
            self.seed,
            self.device,
        )

    def _ensure_built(self) -> None:
        if self.first_order is None or self.second_order is None:
            raise RuntimeError("call .build() before .train() / .evaluate()")

    # ---- train ----------------------------------------------------------

    def train(
        self,
        n_epochs: int | None = None,
        *,
        condition_name: str | None = None,
    ) -> TrainingMetrics:
        """Run the pre-training loop.

        Per epoch :

        1. Generate a fresh batch (default condition : SUPERTHRESHOLD).
        2. Cascade 1st-order forward (1 or N iterations per setting).
        3. ``optimizer_1.zero_grad()``.
        4. If ``setting.second_order`` : cascade 2nd-order forward,
           BCE wager loss, ``loss_2.backward(retain_graph=True)``,
           ``optimizer_2.step()``, ``scheduler_2.step()``.
           Else : no-grad forward (preserves RNG consumption).
        5. CAE (or SimCLR) loss on 1st-order, ``loss_1.backward(retain_graph=True)``,
           ``optimizer_1.step()``, ``scheduler_1.step()``.

        Parameters
        ----------
        n_epochs : int, optional
            Override ``cfg.train.n_epochs`` (paper : 200).
        condition_name : str, optional
            Stimulus regime used at training. Default reads from
            ``env_cfg.pre_training_condition`` (= ``"superthreshold"``).

        Returns
        -------
        TrainingMetrics
            ``losses_1`` (CAE) and ``losses_2`` (BCE wager) per epoch.
        """
        self._ensure_built()
        assert self.first_order is not None
        assert self.second_order is not None
        assert self.optimizer_1 is not None
        assert self.optimizer_2 is not None
        assert self.scheduler_1 is not None
        assert self.scheduler_2 is not None

        n_epochs = n_epochs or self.cfg.train.n_epochs
        condition_name = condition_name or self.cfg.pre_training_condition
        params = ConditionParams(
            random_limit=self.cfg.conditions[condition_name].random_limit,
            baseline=self.cfg.conditions[condition_name].baseline,
            multiplier=self.cfg.conditions[condition_name].multiplier,
        )

        rate_1, iters_1, rate_2, iters_2 = self._cascade_params()
        n_patterns = self.cfg.train.batch_size
        num_units = self.cfg.first_order.input_dim
        loss_kind = self.cfg.first_order_loss.kind

        if loss_kind not in ("cae", "simclr"):
            raise ValueError(f"first_order_loss.kind must be 'cae' or 'simclr'; got {loss_kind!r}")

        losses_1 = np.zeros(n_epochs)
        losses_2 = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            batch = generate_patterns(
                n_patterns=n_patterns,
                num_units=num_units,
                params=params,
                device=self.device,
            )

            # --- Cascade 1st-order forward ---
            h1: torch.Tensor | None = None
            h2: torch.Tensor | None = None
            for _ in range(iters_1):
                h1, h2 = self.first_order(batch.patterns, h1, h2, rate_1)

            assert h1 is not None and h2 is not None

            # --- Two-loss gradient pattern ---
            self.optimizer_1.zero_grad()

            if self.setting.second_order:
                # Cascade 2nd-order
                wager: torch.Tensor | None = None
                comparison: torch.Tensor | None = None
                for _ in range(iters_2):
                    wager, comparison = self.second_order(batch.patterns, h2, comparison, rate_2)
                assert wager is not None

                # eq.5 — wager target = first column of order_2_target (high wager prob)
                target_wager = batch.order_2_target[:, 0].detach()
                loss_2 = wagering_bce_loss(wager.squeeze(-1), target_wager, reduction="sum")
                self.optimizer_2.zero_grad()
                loss_2.backward(retain_graph=True)
                self.optimizer_2.step()
                self.scheduler_2.step()
                losses_2[epoch] = loss_2.item()
            else:
                # No-grad forward — preserves RNG consumption for parity
                with torch.no_grad():
                    comparison_ng: torch.Tensor | None = None
                    for _ in range(iters_2):
                        _, comparison_ng = self.second_order(
                            batch.patterns, h2, comparison_ng, rate_2
                        )

            # --- 1st-order loss (CAE or SimCLR per D11.7) ---
            if loss_kind == "cae":
                # x = stim_present (one-hot target), recons_x = h2
                # Per student blindsight_tmlr.py:pre_train, the CAE
                # reconstruction target is stim_present (not patterns).
                loss_1 = cae_loss(
                    weight=self.first_order.fc1.weight,
                    x=batch.stim_present.detach(),
                    recons_x=h2,
                    hidden=h1,
                    lam=self.cfg.losses.cae_lambda,
                    recon="bce_sum",
                )
            else:  # simclr
                # Build positive pair via bit_flip augmentation (D12.4)
                p = float(
                    self.cfg.get("simclr", {}).get("augmentation", {}).get("bit_flip_prob", 0.1)
                )
                augmented = bit_flip(batch.patterns.detach(), p=p)
                _, h2_aug = self.first_order(augmented, None, None, rate_1)
                # NT-Xent on (h2, h2_aug) — paper "z_i, z_j" as the hidden reps
                loss_1 = simclr_loss(h2, h2_aug, temperature=0.5)

            loss_1.backward(retain_graph=True)
            self.optimizer_1.step()
            self.scheduler_1.step()
            losses_1[epoch] = loss_1.item()

            if (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.info(
                    "epoch %d/%d  loss_1=%.4f  loss_2=%.4f",
                    epoch + 1,
                    n_epochs,
                    losses_1[epoch],
                    losses_2[epoch],
                )

        return TrainingMetrics(losses_1=losses_1, losses_2=losses_2)

    # ---- evaluate -------------------------------------------------------

    def evaluate(
        self,
        n_patterns: int | None = None,
        *,
        conditions: tuple[str, ...] = ("superthreshold", "subthreshold", "low_vision"),
    ) -> EvalMetrics:
        """Compute per-condition discrimination + wager accuracy.

        For each condition :

        - **discrimination_accuracy** : on the stimulus-present half,
          fraction where ``argmax(h2) == argmax(input)`` (matches the
          stimulus index).
        - **wager_accuracy** : on the stimulus-present half, fraction
          where ``(wager > threshold) == (target > threshold)`` with
          per-condition threshold (0.5 super/sub, 0.15 low_vision).

        Parameters
        ----------
        n_patterns : int, optional
            Eval batch size per condition. Defaults to
            ``env_cfg.eval.patterns_number`` (paper : 200).
        conditions : tuple of str, optional
            Which condition names to evaluate.

        Returns
        -------
        EvalMetrics
            Two dicts keyed by condition name.
        """
        self._ensure_built()
        assert self.first_order is not None
        assert self.second_order is not None

        self.first_order.eval()
        self.second_order.eval()

        n_patterns = n_patterns or self.cfg.eval.patterns_number
        rate_1, iters_1, rate_2, iters_2 = self._cascade_params()
        num_units = self.cfg.first_order.input_dim

        metrics = EvalMetrics()

        for condition_name in conditions:
            params = ConditionParams(
                random_limit=self.cfg.conditions[condition_name].random_limit,
                baseline=self.cfg.conditions[condition_name].baseline,
                multiplier=self.cfg.conditions[condition_name].multiplier,
            )
            threshold = float(self.cfg.eval.wager_thresholds[condition_name])

            with torch.no_grad():
                batch = generate_patterns(
                    n_patterns=n_patterns,
                    num_units=num_units,
                    params=params,
                    device=self.device,
                )
                # Cascade 1st-order
                h1: torch.Tensor | None = None
                h2: torch.Tensor | None = None
                for _ in range(iters_1):
                    h1, h2 = self.first_order(batch.patterns, h1, h2, rate_1)
                assert h2 is not None

                # Cascade 2nd-order
                wager: torch.Tensor | None = None
                comparison: torch.Tensor | None = None
                for _ in range(iters_2):
                    wager, comparison = self.second_order(batch.patterns, h2, comparison, rate_2)
                assert wager is not None

                # Split stimulus half (second half)
                delta = n_patterns // 2

                # Discrimination : argmax h2 matches argmax input
                disc_correct = (
                    (h2[delta:].argmax(dim=1) == batch.patterns[delta:].argmax(dim=1))
                    .float()
                    .mean()
                )
                metrics.discrimination_accuracy[condition_name] = disc_correct.item()

                # Wager accuracy : (wager > t) == (target > t)
                wagers_flat = wager[delta:].squeeze(-1).cpu().numpy().flatten()
                # target = first column of order_2_target (high wager prob)
                target_high = batch.order_2_target[delta:, 0].cpu().numpy().flatten()
                tp = np.sum((wagers_flat > threshold) & (target_high > threshold))
                tn = np.sum((wagers_flat < threshold) & (target_high < threshold))
                fp = np.sum((wagers_flat > threshold) & (target_high < threshold))
                fn = np.sum((wagers_flat < threshold) & (target_high > threshold))
                total = tp + tn + fp + fn
                wager_acc = (tp + tn) / total if total > 0 else 0.0
                metrics.wager_accuracy[condition_name] = float(wager_acc)

        self.first_order.train()
        self.second_order.train()
        return metrics
