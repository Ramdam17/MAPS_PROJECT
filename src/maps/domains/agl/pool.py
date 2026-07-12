"""AGL 20-cell replication pool — paper §A.2, Table 10.

After pre-training + reset (:class:`maps.domains.agl.trainer.AGLTrainer`), the
post-pretrain networks are replicated into ``num_networks`` independent cells
(paper: 20). Cells ``[0 : n//2]`` are the **High Awareness** tier (trained many
Grammar-A epochs, e.g. 12); cells ``[n//2 : n]`` are the **Low Awareness** tier
(few epochs, e.g. 3). All cells share the same post-pretrain 2nd-order network
(which already learned "how to judge a reconstruction"); they differ only in how
long their **reset** 1st-order re-learns Grammar-A. That training-budget gap is
what produces the conscious/unconscious wager dissociation.

Port of ``create_networks`` (agl_tmlr.py:814-892) + the aggregation half of
``testing`` (L1150-1270), refactored:
- **in-memory deep-copy** of ``state_dict()`` instead of the student's
  ``torch.save``/``torch.load`` disk round-trip (faster, cleaner, same result);
- Grammar-A training delegated to the shared
  :func:`maps.domains.agl.trainer._run_training_loop`.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §A.2, Table 10.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from maps.domains.agl.data import array_words, target_second
from maps.domains.agl.trainer import (
    AGLTrainer,
    NetworkCell,
    _build_optimizer,
    _run_training_loop,
    calculate_precision,
)

logger = logging.getLogger(__name__)


class AGLNetworkPool:
    """``num_networks`` independent copies of a trainer's post-pretrain nets."""

    def __init__(self, trainer: AGLTrainer, num_networks: int) -> None:
        trainer._ensure_built()
        assert trainer.first_order is not None and trainer.second_order is not None
        self.trainer = trainer
        self.cfg = trainer.cfg
        self.device = trainer.device
        self.cells: list[NetworkCell] = []

        fo_state = copy.deepcopy(trainer.first_order.state_dict())
        so_state = copy.deepcopy(trainer.second_order.state_dict())

        for _ in range(num_networks):
            fo = trainer._make_first_order()
            so = trainer._make_second_order()
            fo.load_state_dict(copy.deepcopy(fo_state))
            so.load_state_dict(copy.deepcopy(so_state))
            # Fresh training-phase optimizers (student create_networks uses
            # lr_training_*; per-cell so pretrain optimizer state never leaks).
            opt1 = _build_optimizer(
                self.cfg.optimizer.name, fo.parameters(), self.cfg.optimizer.lr_training_first_order
            )
            opt2 = _build_optimizer(
                self.cfg.optimizer.name,
                so.parameters(),
                self.cfg.optimizer.lr_training_second_order,
            )
            sched1 = StepLR(
                opt1, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma
            )
            sched2 = StepLR(
                opt2, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma
            )
            self.cells.append(NetworkCell(fo, so, opt1, opt2, sched1, sched2))

        logger.info("Built AGLNetworkPool with %d cells", len(self.cells))

    def __len__(self) -> int:
        return len(self.cells)

    # ---- training a tier -----------------------------------------------

    def train_range(self, start: int, end: int, n_epochs: int) -> dict[str, np.ndarray]:
        """Train cells ``[start:end)`` on Grammar-A for ``n_epochs`` (2nd-order frozen)."""
        rate_1, iters_1, rate_2, iters_2 = self.trainer._cascade_params()
        losses_1, losses_2, precision = _run_training_loop(
            self.cells[start:end],
            n_epochs,
            rate_1=rate_1,
            iters_1=iters_1,
            rate_2=rate_2,
            iters_2=iters_2,
            batch_size=self.cfg.train.batch_size_training,
            cae_lambda=self.cfg.losses.cae_lambda,
            bits_per_letter=self.trainer._bits_per_letter(),
            device=self.device,
        )
        return {"losses_1": losses_1, "losses_2": losses_2, "precision": precision}

    # ---- evaluation ----------------------------------------------------

    def evaluate(self) -> dict[str, dict[str, float]]:
        """Per-cell metrics on a Grammar-A + Grammar-B test batch, aggregated by tier.

        Mirrors ``testing`` (agl_tmlr.py:1150-1270): each cell is evaluated on
        ``len(cells) * factor`` Grammar-A words concatenated with the same number
        of Grammar-B words. Returns mean/std per tier (``high`` = ``[0:n//2]``,
        ``low`` = ``[n//2:]``, ``overall``) for the 1st-order precision and the
        2nd-order wager precision/recall/f1/accuracy.
        """
        n = len(self.cells)
        factor = self.cfg.train.data_factor
        threshold = self.cfg.eval.wager_threshold
        rate_1, iters_1, rate_2, iters_2 = self.trainer._cascade_params()
        bpl = self.trainer._bits_per_letter()

        for cell in self.cells:
            cell.first_order.eval()
            cell.second_order.eval()

        precision_1st = np.zeros(n)
        wager_precision = np.zeros(n)
        wager_recall = np.zeros(n)
        wager_f1 = np.zeros(n)
        wager_accuracy = np.zeros(n)

        for i, cell in enumerate(self.cells):
            with torch.no_grad():
                grammar_a = array_words(2, int(n * factor), device=self.device)
                grammar_b = array_words(3, int(n * factor), device=self.device)
                patterns = torch.cat((grammar_a, grammar_b), dim=0)

                h1 = h2 = None
                for _ in range(iters_1):
                    h1, h2 = cell.first_order(patterns, h1, h2, rate_1)
                comparison = wager = None
                for _ in range(iters_2):
                    wager, comparison = cell.second_order(patterns, h2, comparison, rate_2)
                wager = wager.squeeze()

                precision_1st[i] = calculate_precision(patterns, h2, bits_per_letter=bpl)

                wagers = wager.cpu().numpy().flatten()
                targets = (
                    target_second(patterns, h2, device=self.device).cpu().numpy().flatten() > 0
                ).astype(int)
                tp = int(np.sum((wagers > threshold) & (targets > threshold)))
                tn = int(np.sum((wagers <= threshold) & (targets <= threshold)))
                fp = int(np.sum((wagers > threshold) & (targets <= threshold)))
                fn = int(np.sum((wagers <= threshold) & (targets > threshold)))
                wager_precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                wager_recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                wager_f1[i] = (
                    2
                    * wager_precision[i]
                    * wager_recall[i]
                    / (wager_precision[i] + wager_recall[i])
                    if (wager_precision[i] + wager_recall[i]) > 0
                    else 0.0
                )
                wager_accuracy[i] = (
                    (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                )

        metrics = {
            "precision_1st": precision_1st,
            "wager_precision": wager_precision,
            "wager_recall": wager_recall,
            "wager_f1": wager_f1,
            "wager_accuracy": wager_accuracy,
        }
        half = n // 2
        tiers = {"high": slice(0, half), "low": slice(half, n), "overall": slice(0, n)}
        out: dict[str, dict[str, float]] = {}
        for tier, sl in tiers.items():
            out[tier] = {}
            for key, arr in metrics.items():
                out[tier][key] = float(np.mean(arr[sl]))
                out[tier][f"{key}_std"] = float(np.std(arr[sl]))
        return out
