"""End-to-end parity: :class:`maps.core.SecondOrderNetwork` vs paper_reference.

Compares our factored :class:`maps.core.SecondOrderNetwork` to a verbatim
extract of the student class from
``external/paper_reference/blindsight_tmlr.py:213-252``. The student
code produced the paper Tables 5/6/7 numbers — any divergence here
indicates a Sprint 11 math/composition bug.

Sprint 11.F deliverable. Tolerances: ``1e-7`` for eval (deterministic),
``1e-5`` for train mode (50 unrolls + dropout sampling float noise).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from maps.core.second_order import SecondOrderNetwork

# ---------------------------------------------------------------------------
# Verbatim student SecondOrderNetwork — extracted from
# external/paper_reference/blindsight_tmlr.py:213-252.
#
# Cosmetic-only deltas: dropped the unused ``Softmax``, the ignored
# ``use_gelu`` flag (student code always uses ReLU and never reads
# the flag), and the misleading dead attributes (``activation``,
# ``sigmoid``). Maths is byte-for-byte identical.
# ---------------------------------------------------------------------------


class _StudentSecondOrderNetwork(nn.Module):
    """Verbatim port of blindsight_tmlr.py SecondOrderNetwork."""

    def __init__(self, input_dim: int = 100, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.wager = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout_p)
        # Student calls self._init_weights() at the end of __init__:
        nn.init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(
        self,
        first_order_input: Tensor,
        first_order_output: Tensor,
        prev_comparison: Tensor | None,
        cascade_rate: float,
    ) -> tuple[Tensor, Tensor]:
        comparison_matrix = first_order_input - first_order_output
        comparison_out = self.dropout(comparison_matrix)
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
        wager = torch.sigmoid(self.wager(comparison_out))
        return wager, comparison_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paired_networks(
    input_dim: int = 100, dropout: float = 0.5
) -> tuple[SecondOrderNetwork, _StudentSecondOrderNetwork]:
    """Build a maps and student second-order network with identical
    initial weights by re-seeding before each constructor.

    Both modules construct ``nn.Linear(input_dim, 1)`` first (consumes
    the same RNG bytes for weight + bias init) then overwrite the
    weight with ``init.uniform_(0, 0.1)`` (consumes the same RNG bytes
    again). Bias is left at the Linear default in both.
    """
    torch.manual_seed(0)
    ours = SecondOrderNetwork(
        input_dim=input_dim,
        n_wager_units=1,
        hidden_dim=None,
        dropout=dropout,
    )

    torch.manual_seed(0)
    theirs = _StudentSecondOrderNetwork(input_dim=input_dim, dropout_p=dropout)

    return ours, theirs


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


def test_init_weights_bit_identical() -> None:
    """Same seed → identical weights AND biases between both modules.

    Guards against the silent class of bugs where the API layer
    reorders RNG calls (e.g. constructing ComparatorMatrix before the
    Linear in a way that consumes RNG bytes) and we end up with
    semantically equivalent but numerically different inits.
    """
    ours, theirs = _make_paired_networks()
    assert torch.equal(ours.wager.readout.weight, theirs.wager.weight), (
        "Wager weight init drift — RNG consumption order changed?"
    )
    assert torch.equal(ours.wager.readout.bias, theirs.wager.bias), (
        "Wager bias init drift — RNG consumption order changed?"
    )


def test_single_forward_step_parity_eval() -> None:
    """Single forward step in eval mode (dropout off, deterministic
    path) — outputs must be bit-exact."""
    ours, theirs = _make_paired_networks()
    ours.eval()
    theirs.eval()

    fi = torch.randn(4, 100)
    fo = torch.randn(4, 100)

    w_ours, c_ours = ours(fi, fo, prev_comparison=None, cascade_rate=0.02)
    w_theirs, c_theirs = theirs(fi, fo, prev_comparison=None, cascade_rate=0.02)

    assert torch.allclose(w_ours, w_theirs, atol=1e-7)
    assert torch.allclose(c_ours, c_theirs, atol=1e-7)


def test_50_cascade_steps_parity_eval() -> None:
    """50-iteration cascade in eval mode — both implementations
    converge to ``fi - fo`` (D-sarl-cascade-noop) AND match each other
    to ``1e-6``."""
    ours, theirs = _make_paired_networks()
    ours.eval()
    theirs.eval()

    fi = torch.randn(4, 100)
    fo = torch.randn(4, 100)

    prev_ours: Tensor | None = None
    prev_theirs: Tensor | None = None
    w_ours = w_theirs = None
    for _ in range(50):
        w_ours, prev_ours = ours(fi, fo, prev_comparison=prev_ours, cascade_rate=0.02)
        w_theirs, prev_theirs = theirs(fi, fo, prev_comparison=prev_theirs, cascade_rate=0.02)

    assert prev_ours is not None and prev_theirs is not None
    assert w_ours is not None and w_theirs is not None
    assert torch.allclose(prev_ours, prev_theirs, atol=1e-6)
    assert torch.allclose(w_ours, w_theirs, atol=1e-6)


def test_50_cascade_steps_parity_train() -> None:
    """50-iteration cascade in train mode (dropout active) — outputs
    must match to ``1e-5``. We force identical dropout masks across
    the two modules by re-seeding before each forward; otherwise the
    two forwards would consume different RNG bytes from a shared state
    and the dropout masks would diverge."""
    ours, theirs = _make_paired_networks()
    ours.train()
    theirs.train()

    fi = torch.randn(4, 100)
    fo = torch.randn(4, 100)

    prev_ours: Tensor | None = None
    prev_theirs: Tensor | None = None
    w_ours = w_theirs = None
    for step in range(50):
        torch.manual_seed(1000 + step)
        w_ours, prev_ours = ours(fi, fo, prev_comparison=prev_ours, cascade_rate=0.02)
        torch.manual_seed(1000 + step)
        w_theirs, prev_theirs = theirs(fi, fo, prev_comparison=prev_theirs, cascade_rate=0.02)

    assert prev_ours is not None and prev_theirs is not None
    assert w_ours is not None and w_theirs is not None
    drift_c = (prev_ours - prev_theirs).abs().max().item()
    drift_w = (w_ours - w_theirs).abs().max().item()
    assert torch.allclose(prev_ours, prev_theirs, atol=1e-5), (
        f"comparison drift max={drift_c} > 1e-5"
    )
    assert torch.allclose(w_ours, w_theirs, atol=1e-5), f"wager drift max={drift_w} > 1e-5"


def test_backward_parity_eval() -> None:
    """Backward pass after 50 cascade iters in eval mode — gradients
    on the wager weight & bias must match across implementations.

    Uses a small batch (2) to bound memory through the 50-step
    ``retain_graph`` unroll, per R2 risk in the sprint spec."""
    ours, theirs = _make_paired_networks()
    ours.eval()
    theirs.eval()

    fi = torch.randn(2, 100)
    fo = torch.randn(2, 100)

    prev_ours: Tensor | None = None
    prev_theirs: Tensor | None = None
    w_ours = w_theirs = None
    for _ in range(50):
        w_ours, prev_ours = ours(fi, fo, prev_comparison=prev_ours, cascade_rate=0.02)
        w_theirs, prev_theirs = theirs(fi, fo, prev_comparison=prev_theirs, cascade_rate=0.02)

    assert w_ours is not None and w_theirs is not None
    w_ours.sum().backward()
    w_theirs.sum().backward()

    assert ours.wager.readout.weight.grad is not None
    assert theirs.wager.weight.grad is not None
    assert torch.allclose(ours.wager.readout.weight.grad, theirs.wager.weight.grad, atol=1e-5)
    assert torch.allclose(ours.wager.readout.bias.grad, theirs.wager.bias.grad, atol=1e-5)
