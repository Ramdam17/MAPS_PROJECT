"""AGL pre-training orchestrator.

Ports AGL/AGL_TMLR.py `prepare_pre_training` + `pre_train`
(lines 547-753) into a config-driven class. The goal is that a call to
:meth:`AGLTrainer.pre_train` produces bit-identical loss curves to the
reference script when both are run in ``.eval()`` mode (dropout off) with
the same seed and weights.

Architectural overview
----------------------
Pre-training uses random-grammar words (``GrammarType.RANDOM``) only — the
reference calls ``Array_Words(1, N)`` every epoch. Grammar A and B are held
back for the downstream classification phase (not implemented yet).

Each of the four 2×2 factorial settings maps to cascade rate / meta flag:

===================  =================  ==========
Setting              cascade_rate       meta
===================  =================  ==========
neither              1.0 (α=1, 1 iter)  False
cascade_only         α (α=0.02, 50)     False
second_order_only    1.0                True
both                 α                  True
===================  =================  ==========

Reference reset
---------------
AGL's reference pre_train explicitly restores the first-order network to
its *initial* weights at the end of the loop (`load_state_dict(
initial_first_order_weights)`, L751). This is by design — the downstream
training phase starts from a fresh first-order but keeps the second-order
circuit that learned to wager during pre-training. We replicate that reset
so `AGLTrainer.first_order.state_dict()` after `pre_train()` matches the
reference.

Blindsight does *not* do this reset (see Blindsight_TMLR.py), which is why
`BlindsightTrainer` doesn't either.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR

from maps.components import SecondOrderNetwork, cae_loss, wagering_bce_loss
from maps.experiments.agl.data import (
    BITS_PER_LETTER,
    NUM_INPUT_UNITS,
    GrammarType,
    TrainingBatch,
    generate_batch,
    target_second,
)
from maps.networks import FirstOrderMLP, make_chunked_sigmoid

__all__ = ["AGLSetting", "AGLTrainer"]

log = logging.getLogger(__name__)

_OPTIMIZERS: dict[str, type] = {
    "ADAM": torch.optim.Adam,
    "ADAMW": torch.optim.AdamW,
    "ADAMAX": torch.optim.Adamax,
    "SGD": torch.optim.SGD,
    "RMS": torch.optim.RMSprop,
    "RMSPROP": torch.optim.RMSprop,
}

# RangerVA lives in the optional `torch-optimizer` dep (paper Table 10). Import lazily
# so we can fall back to ADAMAX with a warning if the dep is not installed.
try:
    import torch_optimizer

    _OPTIMIZERS["RANGERVA"] = torch_optimizer.RangerVA
except ImportError:
    log.warning(
        "torch_optimizer not available; 'RANGERVA' optimizer will fall back to "
        "ADAMAX. Install via `uv sync --extra agl` to match paper Table 10."
    )


@dataclass(frozen=True)
class AGLSetting:
    """One entry of the 2×2 factorial ablation (same semantics as Blindsight)."""

    id: str
    label: str
    cascade: bool
    second_order: bool

    @classmethod
    def from_dict(cls, d: DictConfig | dict) -> AGLSetting:
        return cls(
            id=str(d["id"]),
            label=str(d.get("label", d["id"])),
            cascade=bool(d["cascade"]),
            second_order=bool(d["second_order"]),
        )


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    key = name.upper()
    if key == "RANGERVA" and key not in _OPTIMIZERS:
        # Fallback: user asked for RangerVA but torch-optimizer missing.
        log.warning("RangerVA requested but torch-optimizer not installed; falling back to ADAMAX.")
        key = "ADAMAX"
    if key not in _OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {name!r} (options: {sorted(_OPTIMIZERS)})")
    return _OPTIMIZERS[key](params, lr=lr)


class AGLTrainer:
    """Stateful trainer for one (seed, setting) combination.

    Parameters
    ----------
    cfg : DictConfig
        Composed config (from ``load_config('training/agl')``).
    setting : AGLSetting
        Which factorial cell to run.
    device : torch.device | str, default "cpu"
    """

    def __init__(
        self,
        cfg: DictConfig,
        setting: AGLSetting,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.setting = setting
        self.device = torch.device(device)

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
        self._initial_first_order_state: dict | None = None

    def build(self) -> None:
        """Construct networks, optimizers, and schedulers from config."""
        # Sprint-08 D.22b — fail-fast on the D-002 first-order-loss toggle.
        # See docs/reports/sprint-08-d22b-simclr-decision.md.
        from maps.experiments.sarl.training_loop import _check_first_order_loss_kind

        _check_first_order_loss_kind(
            str(self.cfg.get("first_order_loss", {}).get("kind", "cae"))
        )

        fo_cfg = self.cfg.first_order
        so_cfg = self.cfg.second_order
        bits_per_letter = int(self.cfg.get("bits_per_letter", BITS_PER_LETTER))

        self.first_order = FirstOrderMLP(
            input_dim=int(fo_cfg.input_dim),
            hidden_dim=int(fo_cfg.hidden_dim),
            encoder_dropout=float(fo_cfg.encoder_dropout),
            decoder_activation=make_chunked_sigmoid(bits_per_letter),
            weight_init_range=tuple(fo_cfg.weight_init_range),
        ).to(self.device)

        self.second_order = SecondOrderNetwork(
            input_dim=int(so_cfg.input_dim),
            dropout=float(so_cfg.dropout),
            n_wager_units=int(so_cfg.n_wager_units),
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

        # Cache initial weights so `pre_train` can restore them at the end
        # (reference AGL_TMLR.py:751 does this).
        self._initial_first_order_state = copy.deepcopy(self.first_order.state_dict())

    def pre_train(
        self,
        n_epochs: int | None = None,
        *,
        batches: list[TrainingBatch] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the AGL pre-training loop.

        Parameters
        ----------
        n_epochs : int, optional
            Override ``cfg.train.n_epochs_pretrain``.
        batches : list[TrainingBatch], optional
            Pre-generated batches — used by parity tests to eliminate RNG
            synchronization issues. When supplied, ``len(batches)`` sets the
            epoch count and ``n_epochs`` is ignored.

        Returns
        -------
        (losses_1, losses_2) : (np.ndarray, np.ndarray)
            Per-epoch losses. ``losses_2`` is all zeros when the setting has
            ``second_order=False``.

        Side effects
        ------------
        After the loop, the first-order network weights are restored to the
        values captured by ``build()`` (reference behavior, see module docstring).
        The second-order network retains whatever it learned.
        """
        if self.first_order is None:
            raise RuntimeError("Call `.build()` before `.pre_train()`.")

        assert self.second_order is not None
        assert self.optim_1 is not None and self.optim_2 is not None
        assert self.sched_1 is not None and self.sched_2 is not None
        assert self._initial_first_order_state is not None

        if batches is not None:
            n = len(batches)
        else:
            n = int(n_epochs if n_epochs is not None else self.cfg.train.n_epochs_pretrain)
        patterns_number = int(self.cfg.train.batch_size_pretrain)
        num_units = int(self.cfg.first_order.input_dim)
        factor = int(self.cfg.train.get("data_factor", 1))
        lam = float(self.cfg.losses.cae_lambda)

        if num_units != NUM_INPUT_UNITS:
            log.warning(
                "first_order.input_dim=%d ≠ NUM_INPUT_UNITS=%d. "
                "Pattern generation always produces %d-d vectors; "
                "the network will silently truncate/pad. Reference uses 48.",
                num_units,
                NUM_INPUT_UNITS,
                NUM_INPUT_UNITS,
            )

        losses_1 = np.zeros(n)
        losses_2 = np.zeros(n)

        for epoch in range(n):
            if batches is not None:
                batch = batches[epoch]
            else:
                batch = generate_batch(
                    grammar_type=GrammarType.RANDOM,
                    number=patterns_number * factor,
                    device=self.device,
                )

            # First-order cascade pass.
            h1: torch.Tensor | None = None
            h2: torch.Tensor | None = None
            for _ in range(self.cascade_iters):
                h1, h2 = self.first_order(
                    batch.patterns, prev_h1=h1, prev_h2=h2, cascade_rate=self.cascade_rate
                )

            # Reference re-enables autograd on these here; harmless in eval mode.
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
                wager = wager.squeeze()
                wager.requires_grad_(True)

                # Wagering target computed from the current first-order output —
                # 1 iff the top-k output positions exactly match the k active
                # input positions (reference target_second, L421-441).
                target = target_second(batch.patterns, h2)
                loss_2 = wagering_bce_loss(wager, target, reduction="sum").requires_grad_()

                self.optim_2.zero_grad()
                loss_2.backward(retain_graph=True)
                self.optim_2.step()
                self.sched_2.step()
                self.optim_2.zero_grad()
                losses_2[epoch] = float(loss_2.item())
            else:
                with torch.no_grad():
                    for _ in range(self.cascade_iters):
                        _, comparison = self.second_order(
                            batch.patterns, h2, comparison, self.cascade_rate
                        )

            # First-order CAE loss — reconstruction target is the input patterns.
            assert h1 is not None
            W = self.first_order.fc1.weight
            loss_1 = cae_loss(
                W,
                x=batch.patterns.view(-1, num_units),
                recons_x=h2,
                h=h1,
                lam=lam,
                recon="bce_sum",
            )
            loss_1.backward()
            self.optim_1.step()
            self.sched_1.step()
            losses_1[epoch] = float(loss_1.item())

        # Reference behavior — reset first-order to initial state (L751).
        self.first_order.load_state_dict(self._initial_first_order_state)

        return losses_1, losses_2

    def evaluate(
        self,
        *,
        eval_patterns_number: int | None = None,
        threshold: float | None = None,
    ) -> dict[str, float]:
        """Held-out evaluation — ports AGL_TMLR.py `testing()` (L1150).

        Generates a concatenated batch of Grammar A + Grammar B test words
        (size `2 * eval_patterns_number`), runs the cascade forward pass, and
        computes:

        - ``classification_precision`` — per-letter WTA precision on the
          first-order reconstruction (`calculate_metrics` L451).
        - ``wager_accuracy`` — binary classification accuracy of the
          second-order wager against ``target_second``. Only populated when
          ``setting.second_order`` is True.

        Note: the paper's High/Low Awareness split is a **post-hoc seed-pool
        split** (first half of networks → "high", second half → "low") and is
        handled at aggregation time, not here. A single `evaluate()` call
        returns per-network metrics regardless of awareness tier.
        """
        if self.first_order is None or self.second_order is None:
            raise RuntimeError("Call `.build()` before `.evaluate()`.")

        self.first_order.eval()
        self.second_order.eval()

        eval_cfg = self.cfg.get("eval", {}) if hasattr(self.cfg, "get") else {}
        n_eval = int(
            eval_patterns_number
            if eval_patterns_number is not None
            else eval_cfg.get("patterns_number_per_grammar", 20)
        )
        thr = float(
            threshold
            if threshold is not None
            else eval_cfg.get("wager_threshold", self.cfg.train.get("threshold", 0.5))
        )
        bits_per_letter = int(self.cfg.get("bits_per_letter", BITS_PER_LETTER))

        metrics: dict[str, float] = {}

        with torch.no_grad():
            batch_a = generate_batch(grammar_type=GrammarType.A, number=n_eval, device=self.device)
            batch_b = generate_batch(grammar_type=GrammarType.B, number=n_eval, device=self.device)
            patterns = torch.cat((batch_a.patterns, batch_b.patterns), dim=0)

            h1: torch.Tensor | None = None
            h2: torch.Tensor | None = None
            for _ in range(self.cascade_iters):
                h1, h2 = self.first_order(
                    patterns, prev_h1=h1, prev_h2=h2, cascade_rate=self.cascade_rate
                )
            assert h2 is not None

            # Classification precision: WTA on each 6-bit letter chunk, then
            # per-unit precision against the one-hot input.
            pred = torch.zeros_like(h2)
            for row_idx in range(h2.shape[0]):
                for chunk_start in range(0, h2.shape[1], bits_per_letter):
                    chunk = h2[row_idx, chunk_start : chunk_start + bits_per_letter]
                    max_idx = int(torch.argmax(chunk).item())
                    if chunk[max_idx].item() > 0.1:
                        pred[row_idx, chunk_start + max_idx] = 1.0
            tp = float((patterns * pred).sum().item())
            fp = float(((1 - patterns) * pred).sum().item())
            metrics["classification_precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            if self.setting.second_order:
                comparison: torch.Tensor | None = None
                wager: torch.Tensor | None = None
                for _ in range(self.cascade_iters):
                    wager, comparison = self.second_order(
                        patterns, h2, comparison, self.cascade_rate
                    )
                assert wager is not None
                wager = wager.squeeze()
                target = target_second(patterns, h2)
                # Reference flattens both then compares element-wise.
                w = wager.detach().cpu().numpy().flatten()
                t = target.cpu().numpy().flatten()
                pred_bin = (w > thr).astype(int)
                tgt_bin = (t > thr).astype(int)
                metrics["wager_accuracy"] = float((pred_bin == tgt_bin).mean())

        return metrics
