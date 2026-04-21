"""Blindsight pre-training orchestrator.

Ports BLINDSIGHT/Blindsight_TMLR.py `prepare_pre_training` + `pre_train`
(lines 416-630) into a config-driven class. The goal is that a call to
:meth:`BlindsightTrainer.pre_train` produces bit-identical loss curves
to the reference script when both are run in ``.eval()`` mode (dropout
off) with the same seed and weights.

Training logic
--------------
Each of the four 2×2 factorial settings maps to a cascade rate / meta flag:

===================  =================  ==========
Setting              cascade_rate       meta
===================  =================  ==========
neither              1.0 (α=1, 1 iter)  False
cascade_only         α (α=0.02, 50)     False
second_order_only    1.0                True
both                 α                  True
===================  =================  ==========

Per-epoch procedure (reference §pre_train):

1. Generate a fresh batch from the superthreshold condition (pre-training
   always uses the clean regime; testing sweeps all three).
2. Run the first-order cascade loop for ``cascade_iterations`` steps.
3. If ``meta``: run the second-order cascade loop, compute BCE against
   the wagering target, backprop, step optimizer 2 + scheduler 2.
   Else: run a throwaway no-grad second-order forward to preserve the
   reference's RNG consumption pattern.
4. Compute CAE loss on the first-order side, backprop, step optimizer 1
   + scheduler 1.

Notes
-----
- The reference calls ``optimizer_1.zero_grad()`` *before* the second-order
  backward, which means ``loss_2.backward(retain_graph=True)`` accumulates
  gradients into first_order params that are then augmented by loss_1's
  backward before ``optimizer_1.step()`` fires. We replicate this exactly.
- The wager head is 1-unit sigmoid (reference code default, deviation D-001).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR

from maps.components import SecondOrderNetwork, cae_loss, wagering_bce_loss
from maps.experiments.blindsight.data import (
    ConditionParams,
    StimulusCondition,
    generate_patterns,
)
from maps.networks import FirstOrderMLP
from maps.utils.config import load_config

__all__ = ["BlindsightSetting", "BlindsightTrainer"]

log = logging.getLogger(__name__)

_OPTIMIZERS = {
    "ADAM": torch.optim.Adam,
    "ADAMW": torch.optim.AdamW,
    "ADAMAX": torch.optim.Adamax,
    "SGD": torch.optim.SGD,
    "RMS": torch.optim.RMSprop,
    "RMSPROP": torch.optim.RMSprop,
}


@dataclass(frozen=True)
class BlindsightSetting:
    """One entry of the 2×2 factorial ablation.

    Use ``BlindsightSetting.from_dict`` to build from a YAML entry:

        for s in cfg.settings:
            setting = BlindsightSetting.from_dict(s)
    """

    id: str
    label: str
    cascade: bool
    second_order: bool

    @classmethod
    def from_dict(cls, d: DictConfig | dict) -> BlindsightSetting:
        return cls(
            id=str(d["id"]),
            label=str(d.get("label", d["id"])),
            cascade=bool(d["cascade"]),
            second_order=bool(d["second_order"]),
        )


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    key = name.upper()
    if key not in _OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {name!r} (options: {sorted(_OPTIMIZERS)})")
    return _OPTIMIZERS[key](params, lr=lr)


def _condition_params_from_env(env_cfg: DictConfig, which: str) -> ConditionParams:
    p = env_cfg.conditions[which]
    return ConditionParams(
        random_limit=float(p.random_limit),
        baseline=float(p.baseline),
        multiplier=float(p.multiplier),
    )


class BlindsightTrainer:
    """Stateful trainer for one (seed, setting) combination.

    Parameters
    ----------
    cfg : DictConfig
        Composed config (typically from ``load_config('training/blindsight')``).
    setting : BlindsightSetting
        Which factorial cell to run.
    env_cfg : DictConfig, optional
        Pre-loaded ``env/blindsight.yaml`` — useful in tests so the trainer
        can be instantiated with interpolated values already resolved. If
        omitted, loaded on-demand with the main ``cfg`` injected so that
        ``${train.noise_level}`` resolves correctly.
    device : torch.device | str, default "cpu"
    """

    def __init__(
        self,
        cfg: DictConfig,
        setting: BlindsightSetting,
        *,
        env_cfg: DictConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.setting = setting
        self.device = torch.device(device)
        self.env_cfg = env_cfg or self._load_env_cfg(cfg)

        # Cascade schedule — symmetric across both networks (paper convention).
        if setting.cascade:
            self.cascade_rate = float(cfg.cascade.alpha)
            self.cascade_iters = int(cfg.cascade.n_iterations)
        else:
            self.cascade_rate = 1.0
            self.cascade_iters = 1

        self.first_order: FirstOrderMLP | None = None
        self.second_order: SecondOrderNetwork | None = None
        self.optim_1: torch.optim.Optimizer | None = None
        self.optim_2: torch.optim.Optimizer | None = None
        self.sched_1: StepLR | None = None
        self.sched_2: StepLR | None = None

    @staticmethod
    def _load_env_cfg(cfg: DictConfig) -> DictConfig:
        """Load ``env/blindsight.yaml`` with ``cfg.train.noise_level`` in scope.

        The env YAML uses ``${train.noise_level}`` interpolations. We can't
        naively ``load_config("env/blindsight")`` because that file has no
        ``train`` section of its own. Instead we merge it under the main
        ``cfg`` and pull the resolved ``conditions`` tree back out.
        """
        env_raw = load_config("env/blindsight", resolve=False)
        # Merge into main cfg context, then resolve interpolations.
        from omegaconf import OmegaConf

        merged = OmegaConf.merge(cfg, env_raw)
        OmegaConf.resolve(merged)
        return merged  # type: ignore[return-value]

    def build(self) -> None:
        """Construct networks, optimizers, and schedulers from config."""
        # Sprint-08 D.22b — fail-fast on the D-002 first-order-loss toggle.
        # Default `cae` is the paper-faithful-via-student-code path; `simclr`
        # is a reservation guard-rail that raises NotImplementedError.
        # See docs/reports/sprint-08-d22b-simclr-decision.md.
        from maps.experiments.sarl.training_loop import _check_first_order_loss_kind

        _check_first_order_loss_kind(
            str(self.cfg.get("first_order_loss", {}).get("kind", "cae"))
        )

        fo_cfg = self.cfg.first_order
        so_cfg = self.cfg.second_order

        self.first_order = FirstOrderMLP(
            input_dim=int(fo_cfg.input_dim),
            hidden_dim=int(fo_cfg.hidden_dim),
            encoder_dropout=float(fo_cfg.encoder_dropout),
            weight_init_range=tuple(fo_cfg.weight_init_range),
        ).to(self.device)

        self.second_order = SecondOrderNetwork(
            input_dim=int(so_cfg.input_dim),
            dropout=float(so_cfg.dropout),
            n_wager_units=int(so_cfg.n_wager_units),
            hidden_dim=int(so_cfg.get("hidden_dim", 0)),
            weight_init_range=tuple(so_cfg.wager_weight_init_range),
        ).to(self.device)

        opt_cfg = self.cfg.optimizer
        self.optim_1 = _build_optimizer(
            opt_cfg.name, self.first_order.parameters(), lr=float(opt_cfg.lr_first_order)
        )
        self.optim_2 = _build_optimizer(
            opt_cfg.name, self.second_order.parameters(), lr=float(opt_cfg.lr_second_order)
        )

        sch_cfg = self.cfg.scheduler
        self.sched_1 = StepLR(
            self.optim_1, step_size=int(sch_cfg.step_size), gamma=float(sch_cfg.gamma)
        )
        self.sched_2 = StepLR(
            self.optim_2, step_size=int(sch_cfg.step_size), gamma=float(sch_cfg.gamma)
        )

    def _pre_training_params(self) -> ConditionParams:
        """Pick the stimulus condition for pre-training (paper: superthreshold)."""
        name = str(self.env_cfg.pre_training_condition)
        return _condition_params_from_env(self.env_cfg, name)

    def pre_train(
        self,
        n_epochs: int | None = None,
        *,
        condition: StimulusCondition | str | None = None,
        batches: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the pre-training loop.

        Parameters
        ----------
        n_epochs : int, optional
            Override ``cfg.train.n_epochs``. Useful for smoke tests.
        condition : StimulusCondition | str, optional
            Override the default pre-training condition. Rarely needed —
            the paper pre-trains on superthreshold only.
        batches : list[TrainingBatch], optional
            Pre-generated batches — used by parity tests to eliminate
            RNG-synchronization headaches between independent trainers.
            When supplied, ``len(batches)`` determines the epoch count and
            both ``n_epochs`` and ``condition`` are ignored.

        Returns
        -------
        (losses_1, losses_2) : (np.ndarray, np.ndarray)
            Per-epoch losses. ``losses_2`` is all zeros when ``meta=False``.
        """
        if self.first_order is None:
            raise RuntimeError("Call `.build()` before `.pre_train()`.")

        assert self.second_order is not None
        assert self.optim_1 is not None and self.optim_2 is not None
        assert self.sched_1 is not None and self.sched_2 is not None

        if batches is not None:
            n = len(batches)
            params = None
        else:
            n = int(n_epochs if n_epochs is not None else self.cfg.train.n_epochs)
            params = (
                self._pre_training_params()
                if condition is None
                else _condition_params_from_env(self.env_cfg, str(condition))
            )
        patterns_number = int(self.cfg.train.batch_size)
        num_units = int(self.cfg.first_order.input_dim)
        factor = int(self.cfg.train.get("data_factor", 1))
        lam = float(self.cfg.losses.cae_lambda)

        losses_1 = np.zeros(n)
        losses_2 = np.zeros(n)

        for epoch in range(n):
            if batches is not None:
                batch = batches[epoch]
            else:
                assert params is not None
                batch = generate_patterns(
                    params=params,
                    patterns_number=patterns_number,
                    num_units=num_units,
                    factor=factor,
                    device=self.device,
                )

            # First-order cascade pass.
            h1: torch.Tensor | None = None
            h2: torch.Tensor | None = None
            for _ in range(self.cascade_iters):
                h1, h2 = self.first_order(
                    batch.patterns, prev_h1=h1, prev_h2=h2, cascade_rate=self.cascade_rate
                )

            # The reference re-enables autograd on these here; harmless.
            batch.patterns.requires_grad_(True)
            assert h2 is not None
            h2.requires_grad_(True)

            self.optim_1.zero_grad()

            comparison: torch.Tensor | None = None
            if self.setting.second_order:
                # Second-order cascade pass.
                wager: torch.Tensor | None = None
                for _ in range(self.cascade_iters):
                    wager, comparison = self.second_order(
                        batch.patterns, h2, comparison, self.cascade_rate
                    )
                assert wager is not None

                # Wager loss. 1-unit path = student code (sigmoid output + BCE on
                # high-wager target column). 2-unit path = paper eq.3 + eq.5 (raw
                # logits, per-unit BCE-with-logits on 2-D 1-hot target). See D-001.
                if int(self.cfg.second_order.n_wager_units) == 1:
                    target = batch.order_2_target[:, 0]
                    loss_2 = wagering_bce_loss(wager.squeeze(-1), target, reduction="sum")
                else:
                    loss_2 = F.binary_cross_entropy_with_logits(
                        wager, batch.order_2_target, reduction="sum"
                    )

                self.optim_2.zero_grad()
                loss_2.backward(retain_graph=True)  # grads flow into first_order too
                self.optim_2.step()
                self.sched_2.step()
                losses_2[epoch] = float(loss_2.item())
            else:
                # Throwaway pass — preserves reference RNG consumption.
                with torch.no_grad():
                    for _ in range(self.cascade_iters):
                        _, comparison = self.second_order(
                            batch.patterns, h2, comparison, self.cascade_rate
                        )

            # First-order CAE loss — BCE(sum) reconstruction + λ·||J||².
            assert h1 is not None
            W = self.first_order.fc1.weight
            loss_1 = cae_loss(
                W,
                x=batch.stim_present.view(-1, num_units),
                recons_x=h2,
                h=h1,
                lam=lam,
                recon="bce_sum",
            )
            loss_1.backward(retain_graph=True)
            self.optim_1.step()
            self.sched_1.step()
            losses_1[epoch] = float(loss_1.item())

        return losses_1, losses_2

    def evaluate(
        self,
        *,
        eval_patterns_number: int | None = None,
        conditions: tuple[StimulusCondition | str, ...] = (
            StimulusCondition.SUPERTHRESHOLD,
            StimulusCondition.SUBTHRESHOLD,
            StimulusCondition.LOW_VISION,
        ),
    ) -> dict[str, dict[str, float]]:
        """Held-out evaluation — ports the paper's `testing()` loop.

        For each of the three stimulus regimes, generates a fresh 200-trial
        batch (100 noise-only + 100 stimulus-present) and computes:

        - ``discrimination_accuracy`` — fraction of stimulus-present trials
          where ``h2.argmax(dim=1) == stim_present.argmax(dim=1)`` (paper's
          detection metric on the first-order reconstruction).
        - ``wager_accuracy`` — binary classification accuracy of the 2nd-order
          wager output (``high_wager > threshold`` vs. target presence label).
          Only populated when ``setting.second_order`` is True.

        Threshold is 0.5 for superthreshold/subthreshold, 0.15 for low-vision
        (matches the reference `testing()` branch at BLINDSIGHT line 816).

        Returns
        -------
        dict
            Keys are condition names; values are per-condition metric dicts.
        """
        if self.first_order is None or self.second_order is None:
            raise RuntimeError("Call `.build()` before `.evaluate()`.")

        self.first_order.eval()
        self.second_order.eval()

        num_units = int(self.cfg.first_order.input_dim)
        eval_cfg = self.env_cfg.eval
        n_eval = int(
            eval_patterns_number if eval_patterns_number is not None else eval_cfg.patterns_number
        )
        thresholds = eval_cfg.wager_thresholds
        results: dict[str, dict[str, float]] = {}

        with torch.no_grad():
            for cond in conditions:
                cond_name = cond.name.lower() if isinstance(cond, StimulusCondition) else str(cond)
                params = _condition_params_from_env(self.env_cfg, cond_name)
                threshold = float(thresholds[cond_name])

                batch = generate_patterns(
                    params=params,
                    patterns_number=n_eval,
                    num_units=num_units,
                    factor=1,
                    device=self.device,
                )
                # Reference uses delta = int(100 * factor) — the stimulus-present
                # half of the eval batch.
                delta = n_eval // 2

                h1: torch.Tensor | None = None
                h2: torch.Tensor | None = None
                for _ in range(self.cascade_iters):
                    h1, h2 = self.first_order(
                        batch.patterns, prev_h1=h1, prev_h2=h2, cascade_rate=self.cascade_rate
                    )
                assert h2 is not None

                # Discrimination accuracy on the stimulus-present portion.
                # Reference compares argmax(output) to argmax(*input*), not to
                # argmax(stim_present) — stim_present can be all-zero when the
                # drawn stimulus magnitude fell below multiplier/2, but the
                # input still carries the stimulus signal at ``stim_idx``.
                pred_idx = h2[delta:].argmax(dim=1)
                true_idx = batch.patterns[delta:].argmax(dim=1)
                discrimination_acc = float((pred_idx == true_idx).float().mean().item())

                metrics = {"discrimination_accuracy": discrimination_acc}

                if self.setting.second_order:
                    comparison: torch.Tensor | None = None
                    wager: torch.Tensor | None = None
                    for _ in range(self.cascade_iters):
                        wager, comparison = self.second_order(
                            batch.patterns, h2, comparison, self.cascade_rate
                        )
                    assert wager is not None
                    # high-wager > threshold vs. binary presence (target[:,0] is high-wager).
                    # 1-unit path: already sigmoid. 2-unit path: raw logits → sigmoid here
                    # (paper eq.5 per-unit sigmoid at inference). See D-001.
                    if int(self.cfg.second_order.n_wager_units) == 1:
                        high_w = wager[delta:, 0].cpu().numpy()
                    else:
                        high_w = torch.sigmoid(wager[delta:, 0]).cpu().numpy()
                    tgt = batch.order_2_target[delta:, 0].detach().cpu().numpy()
                    pred_bin = (high_w > threshold).astype(int)
                    tgt_bin = (tgt > threshold).astype(int)
                    wager_acc = float((pred_bin == tgt_bin).mean())
                    metrics["wager_accuracy"] = wager_acc

                results[cond_name] = metrics

        return results
