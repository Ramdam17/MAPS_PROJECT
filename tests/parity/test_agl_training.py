"""Parity test: AGLTrainer.training() ≡ reference training loop.

Mirrors the pattern of ``test_agl_pretrain.py`` but targets the Grammar-A
fine-tuning phase. For each factorial setting we :

1. Build the reference FirstOrder/SecondOrder and port FirstOrderMLP/SecondOrderNetwork
   with identical weights (copied ref → port).
2. ``.eval()`` mode on both to kill dropout.
3. Generate N fresh Grammar-A batches with seeded RNG, hand to both.
4. Assert epoch-wise loss_1 / loss_2 / precision curves match to 1e-5.

Student's ``training()`` L969 override (meta=False) is preserved by default
via ``train.train_meta_frozen_in_training=true``. Both sides evaluate
loss_2 for logging, but neither backprops it.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import StepLR

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
from tests.parity._reference_agl_training import reference_agl_training

# Smaller problem than pretrain parity so the test is fast.
N_EPOCHS = 5
NUM_UNITS = 48
HIDDEN = 40
PATTERNS = 15
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
    """Generate `n_epochs` fresh Grammar-A batches with reproducible seed."""
    set_all_seeds(42)
    return [
        generate_batch(grammar_type=GrammarType.A, number=patterns, device="cpu")
        for _ in range(n_epochs)
    ]


@pytest.fixture
def cfg():
    """Parity config: pins optimizer=ADAMAX, second_order.hidden_dim=0 so port
    matches the student reference (no Pasquali hidden layer, ADAMAX)."""
    return load_config(
        "training/agl",
        overrides=[
            f"train.n_epochs_pretrain=1",               # unused here — we test training() only
            f"train.batch_size_training={PATTERNS}",
            f"first_order.input_dim={NUM_UNITS}",
            f"first_order.hidden_dim={HIDDEN}",
            f"second_order.input_dim={NUM_UNITS}",
            "second_order.hidden_dim=0",                # parity w/ reference
            "optimizer.name=ADAMAX",                    # parity w/ reference
            "cascade.alpha=0.2",
            "cascade.n_iterations=5",
            f"losses.cae_lambda={LAM}",
            f"optimizer.lr_training_first_order={LR_1}",
            f"optimizer.lr_training_second_order={LR_2}",
            f"scheduler.step_size={STEP_SIZE}",
            f"scheduler.gamma={GAMMA}",
            "train.train_meta_frozen_in_training=true",
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
def test_agl_training_matches_reference(cfg, setting_id, cascade, second_order):
    setting = AGLSetting(
        id=setting_id, label=setting_id, cascade=cascade, second_order=second_order
    )

    # --- Reference side ---------------------------------------------------
    set_all_seeds(42)
    ref_fo = ReferenceAGLFirstOrderNetwork(
        hidden_units=HIDDEN, data_factor=1, use_gelu=False
    ).eval()
    ref_so = ReferenceAGLSecondOrderNetwork(use_gelu=False, hidden_second=NUM_UNITS).eval()

    # --- Port side: identical weights ------------------------------------
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
        hidden_dim=0,
        weight_init_range=(0.0, 0.1),
    ).eval()
    _copy_first_order(new_fo, ref_fo)
    _copy_second_order(new_so, ref_so)

    # Shared batches.
    batches = _make_batches(N_EPOCHS, PATTERNS)

    # --- Reference run (student training() with meta_frozen=True) --------
    losses_1_ref, losses_2_ref, precision_ref = reference_agl_training(
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
        bits_per_letter=BITS_PER_LETTER,
        meta_frozen_override=True,
    )

    # --- Port run --------------------------------------------------------
    trainer = AGLTrainer(cfg, setting)
    # Bypass build(); install our weight-matched networks directly.
    trainer.first_order = new_fo
    trainer.second_order = new_so
    trainer._initial_first_order_state = copy.deepcopy(new_fo.state_dict())
    # These are built fresh inside trainer.training() per our D.28.b design;
    # no pretrain-phase optimizers needed here.

    losses_1_new, losses_2_new, precision_new = trainer.training(
        n_epochs=N_EPOCHS, batches=batches
    )

    # --- Parity asserts --------------------------------------------------
    np.testing.assert_allclose(
        losses_1_new, losses_1_ref,
        atol=1e-5, err_msg=f"loss_1 mismatch on setting={setting_id}",
    )
    np.testing.assert_allclose(
        losses_2_new, losses_2_ref,
        atol=1e-5, err_msg=f"loss_2 mismatch on setting={setting_id}",
    )
    np.testing.assert_allclose(
        precision_new, precision_ref,
        atol=1e-5, err_msg=f"precision mismatch on setting={setting_id}",
    )
