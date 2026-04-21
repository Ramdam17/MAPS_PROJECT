"""Parity test: AGLTrainer ≡ reference pre_train loop.

For each of the 4 factorial settings, we:

1. Build the reference FirstOrder/SecondOrder networks and the new
   FirstOrderMLP/SecondOrderNetwork (with chunked sigmoid decoder).
2. Copy weights ref → new so both start from identical parameters.
3. Put everything in ``.eval()`` mode to kill dropout. Same tradeoff as
   Blindsight's parity test: eval mode still covers the optimizer /
   scheduler / loss / cascade / target_second wiring — the part most
   likely to regress under refactoring.
4. Generate 5 fixed batches with seeded RNG and hand them to both trainers.
5. Assert epoch-wise loss curves match to 1e-5.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maps.components import SecondOrderNetwork
from maps.experiments.agl import (
    AGLSetting,
    AGLTrainer,
    GrammarType,
    generate_batch,
)
from maps.networks import FirstOrderMLP
from maps.networks.first_order_mlp import make_chunked_sigmoid
from maps.utils import load_config, set_all_seeds
from tests.parity._reference_agl import (
    ReferenceAGLFirstOrderNetwork,
    ReferenceAGLSecondOrderNetwork,
)
from tests.parity._reference_agl_pretrain import reference_agl_pre_train

N_EPOCHS = 5
NUM_UNITS = 48
HIDDEN = 40
PATTERNS = 20
LAM = 1e-4
LR_1 = 0.4
LR_2 = 0.1
STEP_SIZE = 25
GAMMA = 0.98
BITS_PER_LETTER = 6


def _copy_first_order(new: FirstOrderMLP, ref: ReferenceAGLFirstOrderNetwork) -> None:
    with torch.no_grad():
        new.fc1.weight.copy_(ref.fc1.weight)
        new.fc2.weight.copy_(ref.fc2.weight)


def _copy_second_order(new: SecondOrderNetwork, ref: ReferenceAGLSecondOrderNetwork) -> None:
    with torch.no_grad():
        new.wagering_head.wager.weight.copy_(ref.wager.weight)
        new.wagering_head.wager.bias.copy_(ref.wager.bias)


def _make_batches(n_epochs: int, patterns: int):
    """Generate `n_epochs` fresh random-grammar batches with reproducible seed."""
    set_all_seeds(42)
    return [
        generate_batch(grammar_type=GrammarType.RANDOM, number=patterns, device="cpu")
        for _ in range(n_epochs)
    ]


@pytest.fixture
def cfg():
    return load_config(
        "training/agl",
        overrides=[
            f"train.n_epochs_pretrain={N_EPOCHS}",
            f"train.batch_size_pretrain={PATTERNS}",
            f"first_order.input_dim={NUM_UNITS}",
            f"first_order.hidden_dim={HIDDEN}",
            f"second_order.input_dim={NUM_UNITS}",
            "second_order.hidden_dim=0",  # D.28.a: legacy no-hidden wager for parity w/ reference
            "optimizer.name=ADAMAX",      # D.28.a: reference uses ADAMAX; RangerVA is Phase B
            "cascade.alpha=0.2",
            "cascade.n_iterations=5",
            f"losses.cae_lambda={LAM}",
            f"optimizer.lr_first_order={LR_1}",
            f"optimizer.lr_second_order={LR_2}",
            f"scheduler.step_size={STEP_SIZE}",
            f"scheduler.gamma={GAMMA}",
        ],
    )


@pytest.mark.parametrize(
    ("setting_id", "cascade", "second_order"),
    [
        ("neither", False, False),
        ("cascade_only", True, False),
        ("second_order_only", False, True),
        ("both", True, True),
    ],
)
def test_agl_pretrain_matches_reference(cfg, setting_id, cascade, second_order):
    setting = AGLSetting(
        id=setting_id, label=setting_id, cascade=cascade, second_order=second_order
    )

    # --- Reference side ---------------------------------------------------
    set_all_seeds(42)
    ref_fo = ReferenceAGLFirstOrderNetwork(
        hidden_units=HIDDEN, data_factor=1, use_gelu=False
    ).eval()
    ref_so = ReferenceAGLSecondOrderNetwork(use_gelu=False, hidden_second=NUM_UNITS).eval()

    # --- New side: identical weights --------------------------------------
    new_fo = FirstOrderMLP(
        input_dim=NUM_UNITS,
        hidden_dim=HIDDEN,
        encoder_dropout=0.1,
        decoder_activation=make_chunked_sigmoid(BITS_PER_LETTER),
        weight_init_range=(-1.0, 1.0),
    ).eval()
    new_so = SecondOrderNetwork(
        input_dim=NUM_UNITS,
        dropout=0.5,
        n_wager_units=1,
        weight_init_range=(0.0, 0.1),
    ).eval()
    _copy_first_order(new_fo, ref_fo)
    _copy_second_order(new_so, ref_so)

    # Shared batches — generated once, used by both.
    batches = _make_batches(N_EPOCHS, PATTERNS)

    # --- Reference run ----------------------------------------------------
    losses_1_ref, losses_2_ref = reference_agl_pre_train(
        ref_fo,
        ref_so,
        batches=batches,
        meta=second_order,
        cascade_on=cascade,
        cascade_rate=0.2,
        cascade_iterations=5,
        lr_1=LR_1,
        lr_2=LR_2,
        step_size=STEP_SIZE,
        gamma=GAMMA,
        lam=LAM,
        num_units=NUM_UNITS,
    )

    # --- New run ----------------------------------------------------------
    trainer = AGLTrainer(cfg, setting)
    trainer.first_order = new_fo
    trainer.second_order = new_so
    # Re-run `build()`'s optimizer/scheduler steps without re-creating the nets.
    trainer.optim_1 = torch.optim.Adamax(new_fo.parameters(), lr=LR_1)
    trainer.optim_2 = torch.optim.Adamax(new_so.parameters(), lr=LR_2)
    trainer.sched_1 = torch.optim.lr_scheduler.StepLR(
        trainer.optim_1, step_size=STEP_SIZE, gamma=GAMMA
    )
    trainer.sched_2 = torch.optim.lr_scheduler.StepLR(
        trainer.optim_2, step_size=STEP_SIZE, gamma=GAMMA
    )
    # Capture initial weights AFTER the copy, so the reference's "restore to
    # initial" at end of pre_train matches ours.
    import copy as _copy

    trainer._initial_first_order_state = _copy.deepcopy(new_fo.state_dict())

    losses_1_new, losses_2_new = trainer.pre_train(batches=batches)

    # --- Parity asserts ---------------------------------------------------
    np.testing.assert_allclose(
        losses_1_new, losses_1_ref, atol=1e-5, err_msg=f"loss_1 mismatch on setting={setting_id}"
    )
    np.testing.assert_allclose(
        losses_2_new, losses_2_ref, atol=1e-5, err_msg=f"loss_2 mismatch on setting={setting_id}"
    )
