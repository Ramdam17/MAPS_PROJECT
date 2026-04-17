"""First-order autoencoder-style backbone shared by Blindsight + AGL.

A 2-layer MLP (encoder + decoder) with no bias, ReLU on the encoder, optional
dropout, and a configurable decoder activation. The encoder-decoder symmetry
means the hidden representation is tied to a reconstruction objective (see
`maps.components.losses.cae_loss`).

The decoder activation is parametrised because the two reference domains
differ here:

- **Blindsight** applies a single global sigmoid on the 100-d decoder output
  (continuous perceptual signals in [0, 1]).
- **AGL** applies the sigmoid independently on each 6-bit letter chunk
  (discrete symbolic encoding; each chunk approximates a one-hot).

Callers pick the right activation for their domain via `decoder_activation=`.

References
----------
- MAPS paper (Vargas et al., TMLR) §2.2 (Blindsight) and §2.3 (AGL) for the
  two domain-specific variants.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.init as init

from maps.components.cascade import cascade_update


def _global_sigmoid(h: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(h)


def make_chunked_sigmoid(chunk_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a decoder activation that applies sigmoid independently per `chunk_size` units.

    Used by AGL where each 6-bit chunk encodes one letter (bits_per_letter=6).
    """

    def _chunked(h: torch.Tensor) -> torch.Tensor:
        out = h.clone()
        for i in range(0, h.size(-1), chunk_size):
            out[..., i : i + chunk_size] = torch.sigmoid(h[..., i : i + chunk_size])
        return out

    return _chunked


class FirstOrderMLP(nn.Module):
    """Encoder-decoder MLP with cascade-friendly `forward`.

    Parameters
    ----------
    input_dim : int
        Blindsight = 100, AGL = 48. No default because it is always
        domain-dependent and we want explicit instantiation.
    hidden_dim : int, default 40
        Reference AGL value (paper §2.3 default).
    encoder_dropout : float, default 0.1
        Dropout after the ReLU, matching reference code.
    decoder_activation : Callable, default global sigmoid
        Function applied to the decoder output. Use `make_chunked_sigmoid(6)`
        for AGL; keep the default for Blindsight.
    weight_init_range : tuple[float, float], default (-1.0, 1.0)
        Uniform init for both `fc1` and `fc2`. Matches reference code.

    Forward signature
    -----------------
    `forward(x, prev_h1=None, prev_h2=None, cascade_rate=1.0)` returns
    `(h1, h2)`. `prev_h1` is accepted for API symmetry with the reference
    implementation but *not* used — Blindsight/AGL only cascade on the
    decoder output, not on the encoder. Passing a non-None `prev_h1` is a
    no-op; we keep the slot so callers can swap in a richer cascade policy
    later without changing the signature.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 40,
        encoder_dropout: float = 0.1,
        decoder_activation: Callable[[torch.Tensor], torch.Tensor] = _global_sigmoid,
        weight_init_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(encoder_dropout)
        self._decoder_activation = decoder_activation
        init.uniform_(self.fc1.weight, *weight_init_range)
        init.uniform_(self.fc2.weight, *weight_init_range)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.fc1(x.view(-1, self.input_dim))))

    def decoder(
        self,
        z: torch.Tensor,
        prev_h2: torch.Tensor | None,
        cascade_rate: float,
    ) -> torch.Tensor:
        h2 = self._decoder_activation(self.fc2(z))
        return cascade_update(h2, prev_h2, cascade_rate)

    def forward(
        self,
        x: torch.Tensor,
        prev_h1: torch.Tensor | None = None,
        prev_h2: torch.Tensor | None = None,
        cascade_rate: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.encoder(x)
        h2 = self.decoder(h1, prev_h2, cascade_rate)
        return h1, h2
