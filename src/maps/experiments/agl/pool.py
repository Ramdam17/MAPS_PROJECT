"""AGL network pool — 20-copy replication per seed (student L904+ protocol).

The paper's AGL protocol trains **multiple networks per seed** from the same
post-pretrain weights, then splits them into "High Awareness" (12 epochs on
Grammar-A) and "Low Awareness" (3 epochs) tiers for post-hoc evaluation.

Student ``create_networks`` (L814-892) does this via :
1. Save post-pretrain weights to disk.
2. Create ``num_networks=20`` fresh ``(first_order, second_order, optimizer_1,
   optimizer_2, scheduler_1, scheduler_2)`` tuples with the same weights.
3. Return a list of lists: ``networks[i] = [fo, so, crit_1, crit_2, opt_1,
   opt_2, sch_1, sch_2]``.

We port this as a dataclass + class pair:
- :class:`NetworkCell` holds one replica (fo, so, opts, schedulers).
- :class:`AGLNetworkPool` manages the list and delegates training to
  :func:`maps.experiments.agl.trainer._run_training_loop` per cell.

Weight copying uses :func:`copy.deepcopy` on ``state_dict()`` — in-memory,
no disk round-trip like student (which uses ``torch.save(PATH)``).

References
----------
- Paper §A.2 + Table 10 (epochs 12 / 3 for high / low awareness).
- Student ``AGL/AGL_TMLR.py:create_networks`` L814, ``training`` L904,
  ``testing`` L1150.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from maps.components import SecondOrderNetwork
from maps.experiments.agl.data import TrainingBatch
from maps.experiments.agl.trainer import (
    _OPTIMIZERS,
    _build_optimizer,
    _run_training_loop,
    AGLTrainer,
    BITS_PER_LETTER,
)
from maps.networks import FirstOrderMLP

__all__ = ["NetworkCell", "AGLNetworkPool"]

log = logging.getLogger(__name__)


@dataclass
class NetworkCell:
    """One independent (first_order, second_order, opts, schedulers) replica.

    Created from a post-pretrain :class:`AGLTrainer` via
    :meth:`AGLNetworkPool.__init__`. Each cell has its own fresh training-phase
    optimizers (re-built per cell so pretrain state does not leak).
    """

    first_order: FirstOrderMLP
    second_order: SecondOrderNetwork
    optim_1: torch.optim.Optimizer
    optim_2: torch.optim.Optimizer
    sched_1: StepLR
    sched_2: StepLR


class AGLNetworkPool:
    """Pool of ``num_networks`` independent replicas of a post-pretrain trainer.

    Mirrors student ``create_networks`` L814-892 — deep-copies the trainer's
    current first/second-order weights into N cells, each with its own fresh
    training-phase optimizers (``cfg.optimizer.lr_training_*``).

    Intended usage :

        trainer = AGLTrainer(cfg, setting)
        trainer.build()
        trainer.pre_train()                  # pretrain on self.first/second_order
        pool = AGLNetworkPool(trainer, num_networks=20)
        pool.train_range(0, 10, n_epochs=12)   # High Awareness tier
        pool.train_range(10, 20, n_epochs=3)   # Low Awareness tier
        # Next: D.28.d — refactor evaluate to accept pool.

    The trainer's own weights are NOT modified post-pretrain — cells are
    fully independent deep copies.
    """

    def __init__(self, trainer: AGLTrainer, num_networks: int) -> None:
        if trainer.first_order is None or trainer.second_order is None:
            raise RuntimeError(
                "AGLNetworkPool requires a built trainer; call trainer.build() first."
            )

        self.trainer = trainer
        self.num_networks = int(num_networks)
        self.cells: list[NetworkCell] = []

        fo_cfg = trainer.cfg.first_order
        so_cfg = trainer.cfg.second_order
        bits_per_letter = int(trainer.cfg.get("bits_per_letter", BITS_PER_LETTER))
        # Note: bits_per_letter is not needed here — kept as a reminder that cell
        # construction mirrors trainer.build() for the decoder activation.
        _ = bits_per_letter  # unused now, used by decoder in FirstOrderMLP constructor below

        opt_cfg = trainer.cfg.optimizer
        sch_cfg = trainer.cfg.scheduler

        # Snapshot the post-pretrain weights once; cells re-load the same state.
        fo_state = copy.deepcopy(trainer.first_order.state_dict())
        so_state = copy.deepcopy(trainer.second_order.state_dict())

        for i in range(self.num_networks):
            # Rebuild the same architecture as the trainer (matches build()).
            from maps.networks import make_chunked_sigmoid

            fo = FirstOrderMLP(
                input_dim=int(fo_cfg.input_dim),
                hidden_dim=int(fo_cfg.hidden_dim),
                encoder_dropout=float(fo_cfg.encoder_dropout),
                decoder_activation=make_chunked_sigmoid(
                    int(trainer.cfg.get("bits_per_letter", BITS_PER_LETTER))
                ),
                weight_init_range=tuple(fo_cfg.weight_init_range),
            ).to(trainer.device)
            fo.load_state_dict(fo_state)

            so = SecondOrderNetwork(
                input_dim=int(so_cfg.input_dim),
                dropout=float(so_cfg.dropout),
                n_wager_units=int(so_cfg.n_wager_units),
                hidden_dim=int(so_cfg.get("hidden_dim", 0)),
                weight_init_range=tuple(so_cfg.wager_weight_init_range),
            ).to(trainer.device)
            so.load_state_dict(so_state)

            # Fresh per-cell optimizers + schedulers (training-phase LRs).
            o1 = _build_optimizer(
                opt_cfg.name, fo.parameters(), lr=float(opt_cfg.lr_training_first_order)
            )
            o2 = _build_optimizer(
                opt_cfg.name, so.parameters(), lr=float(opt_cfg.lr_training_second_order)
            )
            s1 = StepLR(o1, step_size=int(sch_cfg.step_size), gamma=float(sch_cfg.gamma))
            s2 = StepLR(o2, step_size=int(sch_cfg.step_size), gamma=float(sch_cfg.gamma))

            self.cells.append(
                NetworkCell(first_order=fo, second_order=so, optim_1=o1, optim_2=o2, sched_1=s1, sched_2=s2)
            )

        log.info("AGLNetworkPool: %d cells initialized from post-pretrain weights", self.num_networks)

    def __len__(self) -> int:
        return len(self.cells)

    def train_range(
        self,
        start: int,
        end: int,
        n_epochs: int,
    ) -> dict[str, np.ndarray]:
        """Train cells ``[start:end]`` for ``n_epochs`` on Grammar-A.

        Student protocol (L1499-1505) :
        - High Awareness tier : cells ``[0 : num_networks//2]``, 12 epochs.
        - Low Awareness tier  : cells ``[num_networks//2 : num_networks]``, 3 epochs.

        Returns a dict with arrays of shape ``(end - start, n_epochs)`` for
        ``losses_1``, ``losses_2``, and ``precision`` — one row per cell.
        """
        if not (0 <= start < end <= self.num_networks):
            raise ValueError(
                f"Invalid range [{start}:{end}] for pool of size {self.num_networks}"
            )

        trainer = self.trainer
        cfg = trainer.cfg

        n_cells = end - start
        losses_1 = np.zeros((n_cells, n_epochs))
        losses_2 = np.zeros((n_cells, n_epochs))
        precision = np.zeros((n_cells, n_epochs))

        for idx, cell_idx in enumerate(range(start, end)):
            cell = self.cells[cell_idx]
            l1, l2, p = _run_training_loop(
                first_order=cell.first_order,
                second_order=cell.second_order,
                optim_1=cell.optim_1,
                optim_2=cell.optim_2,
                sched_1=cell.sched_1,
                sched_2=cell.sched_2,
                n_epochs=n_epochs,
                batches=None,
                batch_size=int(cfg.train.batch_size_training),
                num_units=int(cfg.first_order.input_dim),
                factor=int(cfg.train.get("data_factor", 1)),
                lam=float(cfg.losses.cae_lambda),
                bits_per_letter=int(cfg.get("bits_per_letter", BITS_PER_LETTER)),
                meta_frozen=bool(cfg.train.get("train_meta_frozen_in_training", True)),
                setting_second_order=trainer.setting.second_order,
                cascade_iters=trainer.cascade_iters,
                cascade_rate=trainer.cascade_rate,
                device=trainer.device,
            )
            losses_1[idx] = l1
            losses_2[idx] = l2
            precision[idx] = p

        log.info(
            "AGLNetworkPool: trained cells [%d:%d] for %d epochs — mean final precision=%.3f",
            start,
            end,
            n_epochs,
            float(precision[:, -1].mean()) if n_epochs > 0 else 0.0,
        )

        return {"losses_1": losses_1, "losses_2": losses_2, "precision": precision}
