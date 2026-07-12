"""MARL meta (wager) networks — MAPS §5 (M-C2/M-C3).

Ported from ``onpolicy/algorithms/r_mappo/algorithm/r_actor_critic_meta.py`` +
``utils/rnn_meta.py``. These produce the **wager** the trainer supervises with a
BCE against ``reward>0`` (M-C2). Key faithful properties of the paper's code:

- The meta networks do **NOT** select env actions (only the baseline R_Actor does)
  — the meta path is a passive wager read-out.
- ``r_mappo.evaluate_actions_meta`` uses ``actor_meta`` for BOTH the actor- and
  critic-side wager losses, so **critic_meta never receives gradients** (its
  optimizer step is a source no-op). R_Critic_Meta is therefore structural-only
  (dead weight), ported for fidelity.
- ``RNNLayer_Meta`` carries an internal wager head that is unused on the
  ``wager=False`` path R_Actor_Meta.evaluate_actions takes; the wager comes from
  the separate ``SecondOrderNetwork``.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from maps.domains.marl.encoder import CNNBase
from maps.domains.marl.util import check


class SecondOrderNetwork(nn.Module):
    """Comparator + wager head (hidden-sized). Verbatim r_actor_critic_meta.py:17-42."""

    def __init__(self, num_linear_units: int) -> None:
        super().__init__()
        self.comparison_layer = nn.Linear(num_linear_units, num_linear_units)
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)
        nn.init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        nn.init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):
        comparison_out = self.dropout(
            torch.nn.functional.relu(self.comparison_layer(comparison_matrix))
        )
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
        return self.wager(comparison_out), comparison_out


class RNNLayer_Meta(nn.Module):  # noqa: N801 (source class name)
    """GRU + dual-cascade meta RNN. Verbatim rnn_meta.py:7-115 (wager=False path used)."""

    def __init__(
        self, inputs_dim: int, outputs_dim: int, recurrent_n: int, use_orthogonal: bool
    ) -> None:
        super().__init__()
        self._recurrent_N = recurrent_n
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_n)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                (nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_)(param)
        self.norm = nn.LayerNorm(outputs_dim)
        self.wager = nn.Linear(outputs_dim, 2)  # internal wager head (unused on wager=False path)
        self.sigmoid = torch.sigmoid

    def _gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1))
                .transpose(0, 1)
                .contiguous(),
            )
            return x.squeeze(0), hxs.transpose(0, 1)
        batch_num = hxs.size(0)
        episode_len = int(x.size(0) / batch_num)
        x = x.view(episode_len, batch_num, x.size(1))
        masks = masks.view(episode_len, batch_num)
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
        has_zeros = (
            [has_zeros.item() + 1] if has_zeros.dim() == 0 else (has_zeros + 1).numpy().tolist()
        )
        has_zeros = [0, *has_zeros, episode_len]
        hxs = hxs.transpose(0, 1)
        outputs = []
        for i in range(len(has_zeros) - 1):
            temp = (
                hxs * masks[has_zeros[i]].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)
            ).contiguous()
            rnn_scores, hxs = self.rnn(x[has_zeros[i] : has_zeros[i + 1]], temp)
            outputs.append(rnn_scores)
        x = torch.cat(outputs, dim=0).reshape(episode_len * batch_num, -1)
        return x, hxs.transpose(0, 1)

    def forward(self, x, hxs, masks, prev_h1, prev_h2, cascade_rate1, cascade_rate2, wager=False):
        x, hxs = self._gru(x, hxs, masks)
        if wager:
            x = self.norm(x)
            output_cascade2 = (
                x if prev_h2 is None else cascade_rate2 * x + (1 - cascade_rate2) * prev_h2
            )
            return self.sigmoid(self.wager(output_cascade2)), hxs, output_cascade2
        output_cascade1 = (
            x if prev_h1 is None else cascade_rate1 * x + (1 - cascade_rate1) * prev_h1
        )
        return self.norm(output_cascade1), hxs, output_cascade1


class R_Actor_Meta(nn.Module):  # noqa: N801 (source class name)
    """Meta actor — produces the wager (does NOT act). Port of R_Actor_Meta."""

    def __init__(
        self,
        obs_shape,
        action_space,
        *,
        hidden_size: int,
        recurrent_n: int,
        cascade_iterations_1: int,
        cascade_iterations_2: int,
        use_orthogonal: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cascade_one = cascade_iterations_1
        self.cascade_two = cascade_iterations_2
        self.cascade_rate_one = float(1.0 / cascade_iterations_1)
        self.cascade_rate_two = float(1.0 / cascade_iterations_2)
        self.device = torch.device(device)
        self.layer_input = nn.Linear(hidden_size, hidden_size)
        self.layer_output = nn.Linear(hidden_size, hidden_size)
        self.second_order = SecondOrderNetwork(hidden_size)
        self.base = CNNBase(obs_shape, hidden_size, use_orthogonal, use_relu=True)
        self.rnn = RNNLayer_Meta(hidden_size, hidden_size, recurrent_n, use_orthogonal)
        # act head exists in the source but the meta actor does not select env actions.
        self.to(self.device)

    def evaluate_actions(
        self, obs, rnn_states, action=None, masks=None, available_actions=None, active_masks=None
    ) -> Tensor:
        """Return the wager (B, 2) — the trainer's meta target signal. Verbatim path."""
        obs = check(obs).to(self.device).float()
        rnn_states = check(rnn_states).to(self.device).float()
        masks = check(masks).to(self.device).float()

        actor_features = self.layer_input(self.base(obs))
        rnn_states = self.layer_input(rnn_states)
        initial_states = actor_features
        features_out, rnn_out, output_cascade1 = actor_features, rnn_states, None
        for _ in range(self.cascade_one):
            features_out, rnn_out, output_cascade1 = self.rnn(
                features_out,
                rnn_out,
                masks,
                output_cascade1,
                None,
                self.cascade_rate_one,
                self.cascade_rate_two,
                wager=False,
            )
        comparison_matrix = initial_states - features_out
        prev_comparison = None
        wager = None
        for _ in range(self.cascade_two):
            wager, prev_comparison = self.second_order(
                comparison_matrix, prev_comparison, self.cascade_rate_two
            )
        return wager


class R_Critic_Meta(nn.Module):  # noqa: N801 (source class name)
    """Meta critic — DEAD WEIGHT (M-C3): never receives gradients in the source
    (evaluate_actions_meta uses actor_meta both sides). Ported structurally only."""

    def __init__(
        self,
        share_obs_shape,
        *,
        hidden_size: int,
        recurrent_n: int,
        cascade_iterations_1: int,
        cascade_iterations_2: int,
        use_orthogonal: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.layer_input = nn.Linear(hidden_size, hidden_size)
        self.second_order = SecondOrderNetwork(hidden_size)
        self.base = CNNBase(share_obs_shape, hidden_size, use_orthogonal, use_relu=True)
        self.rnn = RNNLayer_Meta(hidden_size, hidden_size, recurrent_n, use_orthogonal)
        self.to(self.device)
