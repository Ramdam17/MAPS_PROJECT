"""Minimal student-code reference for MARL parity testing (E.13).

Vendors **only** the leaf modules we need to reproduce the student's
MAPPO forward pass bit-exactly. The full student tree (SCOFF/RIM/MLP/PopArt)
is NOT imported — only the clean no-attention path that our port targets.

Source : ``external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/``
(at git SHA of E.1 restoration). Changes vs source :

- Stripped ``from .modularity import SCOFF`` and ``from .rim_cell import RIM``
  (attention paths we dropped per E.5).
- Stripped ``MLPBase`` (1D obs path ; MeltingPot is always RGB).
- Stripped ``PopArt`` (use_popart=False per E.5).
- Stripped ``FourierPositionEncoding`` / ``ResidualBlock`` / ``Encoder``
  (perceiver code unused by MAPPO).

What is preserved verbatim (ops identical to source) :
- :mod:`util` : ``init``, ``check``, ``calculate_conv_params``.
- :mod:`cnn` : ``Flatten``, ``CNNLayer``, ``CNNBase``.
- :mod:`rnn` : ``RNNLayer``.
- :mod:`distributions` : ``FixedCategorical``, ``Categorical``.
- :mod:`act` : ``ACTLayer`` (Discrete branch only).

Use only from ``tests/parity/test_marl_forward_parity.py``.
"""
