"""First-order MLP — encoder/decoder backbone shared by Blindsight & AGL.

MAPS paper §2.2 (Blindsight Table 9) and §2.3 (AGL Table 10).

Architecture
------------
::

    x → Linear(input_dim, hidden_dim, bias=False) → ReLU → Dropout
      → Linear(hidden_dim, input_dim, bias=False) → decoder_activation

No biases anywhere. Weight init uniform ``(-1, 1)``. The cascade
(:func:`maps.core.cascade.cascade_update`) is applied **on the decoder
output (h2) only** — the encoder hidden ``h1`` is recomputed each step.

Domain differences
------------------
- **Blindsight** : ``decoder_activation = global_sigmoid`` (one sigmoid
  on the full 100-dim output).
- **AGL** : ``decoder_activation = make_chunked_sigmoid(6)`` — sigmoid
  applied per 6-bit chunk (each chunk encodes one letter from an
  8-letter, 6-bit alphabet).

SARL does **not** use this module (it has a convolutional
``SarlQNetwork`` in :mod:`maps.domains.sarl.model`).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §2.2, §2.3, Tables 9-10.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from maps.core.cascade import cascade_update


def global_sigmoid(x: Tensor) -> Tensor:
    """Element-wise sigmoid on the full last dimension.

    Used as the Blindsight decoder activation — outputs probabilities
    in ``[0, 1]`` per unit.
    """
    return torch.sigmoid(x)


def make_chunked_sigmoid(chunk_size: int) -> Callable[[Tensor], Tensor]:
    r"""Factory : returns a sigmoid that operates on ``chunk_size``-wide
    chunks of the last dimension independently.

    Mathematically identical to :func:`global_sigmoid` (sigmoid is
    element-wise — chunking doesn't change values). The chunking
    structure is kept for **semantic intent** : in AGL the input
    encodes letters as 6 bits each, and the chunked operator
    documents that the decoder treats each letter independently.

    The student code modifies the tensor in-place via slicing
    assignment. This implementation builds the output via
    :func:`torch.cat` to avoid the PyTorch autograd pitfalls that can
    arise from in-place modification of a tensor that is part of an
    upstream computation graph.

    Parameters
    ----------
    chunk_size : int
        Width of each chunk along the last dimension. Must be > 0 and
        divide the last-dim size of the input at call time (caller's
        responsibility).

    Returns
    -------
    Callable[[Tensor], Tensor]
        A closure that applies chunked sigmoid.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}")

    def chunked_sigmoid(x: Tensor) -> Tensor:
        chunks = [
            torch.sigmoid(x[..., i : i + chunk_size]) for i in range(0, x.shape[-1], chunk_size)
        ]
        return torch.cat(chunks, dim=-1)

    return chunked_sigmoid


class FirstOrderMLP(nn.Module):
    r"""Encoder-decoder MLP backbone (Blindsight & AGL).

    Parameters
    ----------
    input_dim : int
        Input dimensionality (Blindsight: 100, AGL: 48).
    hidden_dim : int
        Encoder hidden width. Blindsight default: 40 (D12.3 — student
        ``main()`` value, RG-002 H5). AGL: 40 (paper Table 10).
    decoder_activation : Callable[[Tensor], Tensor]
        Activation applied to the decoder linear output. Use
        :func:`global_sigmoid` for Blindsight, :func:`make_chunked_sigmoid`
        ``(6)`` for AGL.
    encoder_dropout : float, optional
        Dropout probability on the encoder hidden (default ``0.1``
        per paper).
    weight_init_range : tuple of two floats, optional
        Lower and upper bound for uniform weight init (default
        ``(-1.0, 1.0)``).

    Attributes
    ----------
    fc1 : torch.nn.Linear
        Encoder ``input_dim → hidden_dim``, ``bias=False``.
    fc2 : torch.nn.Linear
        Decoder ``hidden_dim → input_dim``, ``bias=False``.
    dropout : torch.nn.Dropout
        Encoder dropout (paper §2.2, p=0.1).
    decoder_activation : Callable
        Stored for use in :meth:`forward`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        decoder_activation: Callable[[Tensor], Tensor],
        *,
        encoder_dropout: float = 0.1,
        weight_init_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError(f"input_dim and hidden_dim must be ≥ 1, got {input_dim}, {hidden_dim}")
        a, b = weight_init_range
        if not a < b:
            raise ValueError(
                f"weight_init_range must be (lower, upper) with lower < upper, "
                f"got {weight_init_range}"
            )

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(p=encoder_dropout)
        self.decoder_activation = decoder_activation

        nn.init.uniform_(self.fc1.weight, a=a, b=b)
        nn.init.uniform_(self.fc2.weight, a=a, b=b)

    def forward(
        self,
        x: Tensor,
        prev_h1: Tensor | None = None,
        prev_h2: Tensor | None = None,
        cascade_rate: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        r"""Forward pass with optional cascade on the decoder output.

        Parameters
        ----------
        x : torch.Tensor
            Input batch, shape ``(N_batch, input_dim)``.
        prev_h1 : torch.Tensor or None, optional
            **Accepted but unused.** Kept for API symmetry with the
            student trainer (which passes both ``prev_h1`` and
            ``prev_h2`` even though only the latter cascades). The
            paper applies the cascade on the "last hidden layer" which
            here is the decoder output. Kept as a slot in case a
            future cascade policy needs encoder-side accumulation.
        prev_h2 : torch.Tensor or None, optional
            Previous decoder output. ``None`` at bootstrap (t=0) — in
            that case :func:`cascade_update` returns the fresh value
            unchanged.
        cascade_rate : float, optional
            Integration rate ``α ∈ (0, 1]``. Default ``1.0`` ≡ "no
            cascade" (collapse to fresh decoder output). Paper uses
            ``α = 0.02`` over 50 iterations.

        Returns
        -------
        h1 : torch.Tensor
            Post-dropout encoder hidden, shape ``(N_batch, hidden_dim)``.
        h2 : torch.Tensor
            Cascaded decoder output, shape ``(N_batch, input_dim)``.
        """
        # Encoder — not cascaded (paper: cascade on last hidden, i.e. decoder)
        h1_raw = F.relu(self.fc1(x))
        h1 = self.dropout(h1_raw)
        del prev_h1  # explicit no-op — see docstring

        # Decoder + cascade on h2
        h2_raw = self.fc2(h1)
        h2_activated = self.decoder_activation(h2_raw)
        h2 = cascade_update(h2_activated, prev_h2, cascade_rate)

        return h1, h2
