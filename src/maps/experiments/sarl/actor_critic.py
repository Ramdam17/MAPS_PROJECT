"""Actor-Critic with eligibility traces (AC(λ)) — paper Setting 7 baseline.

Ports ``external/paper_reference/sarl_ac_lambda.py`` (Young & Tian 2019,
*MinAtar: An Atari-inspired testbed for thorough and reproducible
reinforcement learning experiments*, arXiv:1903.03176, Algorithm 1) into a
config-driven trainer class.

This is the **paper Setting 7 (ACB = Actor-Critic Baseline)** referenced in
Tables 6 (SARL) and 7 (MARL) — a structurally different algorithm from the
DQN+meta+cascade pipeline used in Settings 1-6. ACB has neither cascade nor
the second-order network ; it is a competitive non-MAPS baseline against
which MAPS' improvement is claimed.

Algorithmic distinctives (preserved literally from the reference)
---------------------------------------------------------------
- **Online actor-critic**, no experience replay buffer.
- **Eligibility traces** with ``λ=0.8, γ=0.99`` (TD(λ) replacing-trace style).
- **Custom RMSprop with initialization debiasing** :
  ``MSG_t = γ_rms · MSG_{t-1} + (1 - γ_rms) · grad²``
  ``param += α · grad / √(MSG_t / (1 - γ_rms^(t+1)) + ε_rms)``
  The ``(1 - γ_rms^(t+1))`` debiasing term is *not* in ``torch.optim.RMSprop``
  — using PyTorch's optimizer would diverge from the reference. We replicate
  the manual update inside :meth:`ACBTrainer._update`.
- **Entropy bonus** ``β=0.01`` on the policy distribution.
- **SiLU / dSiLU activations** (Elfwing et al. 2018).
- **Numerical floor** ``MIN_DENOM=1e-4`` inside ``log(π + MIN_DENOM)``.
  Not in paper Table 11 ; logged in ``deviations.md`` as ``D-sarl-acb-min-denom``.

Reference
---------
- Young, K. & Tian, T. (2019). *MinAtar : An Atari-Inspired Testbed for Thorough
  and Reproducible Reinforcement Learning Experiments.* arXiv:1903.03176. §5.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning : An Introduction.*
  §13.5 (actor-critic with eligibility traces).
- Vargas et al. (2025), MAPS TMLR submission §3.2 and Tables 6/7.
- Vendored reference : ``external/paper_reference/sarl_ac_lambda.py``.
"""

from __future__ import annotations

import json
import logging
import time
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ACBConfig", "ACBTrainer", "ACNetwork", "Transition"]

log = logging.getLogger(__name__)


# ── Paper-locked hyperparameters (Young & Tian 2019, reference constants) ────

DEFAULT_ALPHA: float = 0.00048828125  # 1 / 2048, AC(λ) step size
DEFAULT_LAMBDA: float = 0.8            # trace decay
DEFAULT_GAMMA: float = 0.99            # discount factor
DEFAULT_BETA: float = 0.01             # entropy bonus weight
DEFAULT_GAMMA_RMS: float = 0.999       # RMSprop EMA on squared grad
DEFAULT_EPS_RMS: float = 0.0001        # RMSprop epsilon
DEFAULT_MIN_DENOM: float = 0.0001      # numerical floor in log(π) — D-sarl-acb-min-denom


Transition = namedtuple("Transition", ["state", "last_state", "action", "reward", "is_terminal"])


# ── SiLU activations (Elfwing et al. 2018) ──────────────────────────────────


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) = x · σ(x). Identical to ``torch.nn.functional.silu``."""
    return x * torch.sigmoid(x)


def dsilu(x: torch.Tensor) -> torch.Tensor:
    """Derivative-of-SiLU : σ(x) · (1 + x · (1 - σ(x))).

    The two ``torch.sigmoid(x)`` calls are *intentional* — they mirror the
    reference (``external/paper_reference/sarl_ac_lambda.py`` lines 42-43,
    ``dSiLU = lambda x: torch.sigmoid(x)*(1+x*(1-torch.sigmoid(x)))``).
    Caching ``s = sigmoid(x)`` as a single node changes the autograd graph
    structure, which then changes the floating-point order of gradient
    accumulation in ``param.grad`` — divergence ≈ 1 ULP float32 per backward
    pass, observed empirically against the reference. Do NOT refactor to
    avoid the double sigmoid call without re-running the parity test.
    """
    return torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))


# ── ACNetwork ────────────────────────────────────────────────────────────────


class ACNetwork(nn.Module):
    """Conv(16) → FC(128) → (policy softmax, value linear).

    Mirrors the reference ``ACNetwork`` (lines 60-91 of vendored ref) — conv
    with 16 filters (vs DQN's 64, paper §5 says "a quarter of the original
    DQN"), FC with 128 rectified units, then two heads : ``policy``
    (``num_actions`` outputs, softmax) and ``value`` (single linear output).

    Activations are SiLU on the conv output and dSiLU on the FC output
    (faithful to the reference — these are the activations Elfwing recommends
    for value-based RL).

    Parameters
    ----------
    in_channels : int
        Number of input channels (varies by MinAtar game — call
        ``env.state_shape()[2]`` to get it).
    num_actions : int
        Number of valid actions for the environment.
    """

    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        # Reference uses a single conv layer ; output is 10-2=8 per spatial
        # dim ⇒ 8 · 8 · 16 = 1024 features into the FC layer.
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size: int, kernel_size: int = 3, stride: int = 1) -> int:
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.policy = nn.Linear(in_features=128, out_features=num_actions)
        self.value = nn.Linear(in_features=128, out_features=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = silu(self.conv(x))
        x = dsilu(self.fc_hidden(x.view(x.size(0), -1)))
        return F.softmax(self.policy(x), dim=1), self.value(x)


# ── Config dataclass ─────────────────────────────────────────────────────────


@dataclass
class ACBConfig:
    """Run-time configuration for one (game, seed) ACB cell.

    All algorithmic constants default to the Young & Tian 2019 paper values
    as copied from the reference. Override only when a deviation is logged
    in ``docs/reproduction/deviations.md``.

    Parameters
    ----------
    game : str
        MinAtar environment id — one of ``space_invaders, breakout, seaquest,
        asterix, freeway``.
    seed : int
        Master seed. Set via :func:`maps.utils.seeding.set_all_seeds` at the
        trainer's entry point.
    num_frames : int
        Training budget in environment steps. Paper Table 11 says 500_000.
    output_dir : Path
        Where ``summary.json`` and ``.pt`` checkpoints are written.
    alpha, lambda_, gamma, beta, gamma_rms, eps_rms, min_denom : float
        Paper-locked algorithmic constants — see module docstring.
    validation_every_episodes : int
        Run validation every N training episodes. Reference uses 500 for
        most games and 15 for freeway (long episodes).
    validation_episodes : int
        Number of evaluation episodes per validation point. Reference uses 2.
    device : str
        ``cpu`` (default — MinAtar is small enough), ``cuda``, or ``cuda:N``.
    log_every_episodes : int
        Verbose log cadence. Set to 0 to disable progress lines.
    """

    game: str
    seed: int
    num_frames: int
    output_dir: Path

    alpha: float = DEFAULT_ALPHA
    lambda_: float = DEFAULT_LAMBDA
    gamma: float = DEFAULT_GAMMA
    beta: float = DEFAULT_BETA
    gamma_rms: float = DEFAULT_GAMMA_RMS
    eps_rms: float = DEFAULT_EPS_RMS
    min_denom: float = DEFAULT_MIN_DENOM

    validation_every_episodes: int = 500
    validation_episodes: int = 2
    device: str = "cpu"
    log_every_episodes: int = 100

    @classmethod
    def for_game(cls, game: str, **kwargs) -> ACBConfig:
        """Build a config with per-game defaults applied.

        Freeway has very long episodes (one game = many minutes of env time),
        so the reference validates every 15 episodes instead of 500. Apply
        the override unconditionally — using ``setdefault`` here was a bug
        because ``run_sarl.py`` always passes ``validation_every_episodes``
        from the YAML, masking the per-game default. The 2026-05-19 ACB
        production run hit n_validation_points=0 on all 3 freeway seeds for
        that reason.
        """
        if game == "freeway":
            kwargs["validation_every_episodes"] = 15
        return cls(game=game, **kwargs)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_state(s, device: torch.device) -> torch.Tensor:
    """Convert MinAtar state (H, W, C numpy) to a (1, C, H, W) float tensor.

    Mirrors the reference ``get_state`` exactly.
    """
    return torch.tensor(s, device=device).permute(2, 0, 1).unsqueeze(0).float()


def _world_step(s: torch.Tensor, env, network: ACNetwork) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One env step under the current policy. Mirrors reference ``world_dynamics``."""
    with torch.no_grad():
        action = torch.multinomial(network(s)[0], 1)[0]
    reward, terminated = env.act(action)
    s_prime = _get_state(env.state(), s.device)
    return (
        s_prime,
        action,
        torch.tensor([[reward]], device=s.device).float(),
        torch.tensor([[terminated]], device=s.device),
    )


def _validation(env, network: ACNetwork, num_episodes: int, device: torch.device) -> tuple[float, float]:
    """Average + std of returns over ``num_episodes`` greedy episodes.

    Reference still samples from π during validation (no argmax) — we mirror
    that exactly. Returns are summed undiscounted.
    """
    network.eval()
    returns = []
    for _ in range(num_episodes):
        env.reset()
        s = _get_state(env.state(), device)
        G = 0.0
        terminated = False
        while not terminated:
            with torch.no_grad():
                action = torch.multinomial(network(s)[0], 1)[0]
            reward, terminated = env.act(action)
            G += reward
            if not terminated:
                s = _get_state(env.state(), device)
        returns.append(G)
    network.train()
    return float(np.mean(returns)), float(np.std(returns))


# ── Trainer ──────────────────────────────────────────────────────────────────


class ACBTrainer:
    """Stateful AC(λ) trainer — one (game, seed) cell.

    Usage
    -----
    >>> cfg = ACBConfig.for_game("breakout", seed=42, num_frames=500_000,
    ...                          output_dir=Path("./outputs/sarl/breakout/setting-7/seed-42"))
    >>> from minatar import Environment
    >>> trainer = ACBTrainer(cfg, Environment("breakout"))
    >>> summary = trainer.train()

    The trainer writes ``summary.json``, ``policy_value.pt``, ``returns.npy``
    and ``validation_returns.npy`` under ``cfg.output_dir`` at the end of the
    run. Intermediate checkpointing is not implemented (the reference's
    ``store_intermediate_result`` flag is dropped — Phase γ runs are short
    enough to not need it).
    """

    def __init__(self, cfg: ACBConfig, env) -> None:
        self.cfg = cfg
        self.env = env
        self.device = torch.device(cfg.device)

        in_channels = env.state_shape()[2]
        num_actions = env.num_actions()
        self.network = ACNetwork(in_channels, num_actions).to(self.device)

        # Per-parameter buffers : eligibility traces, gradient snapshots,
        # mean-squared-gradient EMA for the custom RMSprop.
        self.traces: list[torch.Tensor] = [
            torch.zeros_like(p, device=self.device) for p in self.network.parameters()
        ]
        self.grads: list[torch.Tensor] = [
            torch.zeros_like(p, device=self.device) for p in self.network.parameters()
        ]
        self.msg: list[torch.Tensor] = [
            torch.zeros_like(p, device=self.device) for p in self.network.parameters()
        ]

    def _update(self, sample: Transition, time_step: int) -> None:
        """One AC(λ) update — mirrors reference ``train()`` literally.

        Two-pass gradient computation :
        1. Compute ``trace_potential = V_curr + 0.5 · log(π[action] + MIN_DENOM)``,
           backprop into ``self.grads`` (snapshot).
        2. If a previous state exists : compute entropy bonus backward,
           combine with snapshot to form the AC(λ) gradient
           ``grad = trace · δ + β · ∂H/∂θ``, and update params via the
           custom debiased RMSprop.
        3. Always update eligibility traces : ``trace ← λ·γ·trace + grad_snapshot``.
        """
        cfg = self.cfg
        net = self.network

        last_state = sample.last_state
        state = sample.state
        action = sample.action
        reward = sample.reward
        is_terminal = sample.is_terminal

        pi, v_curr = net(state)

        trace_potential = v_curr + 0.5 * torch.log(pi[0, action] + cfg.min_denom)
        entropy = -torch.sum(torch.log(pi + cfg.min_denom) * pi)

        # Pass 1 : snapshot ∂trace_potential/∂θ into self.grads.
        net.zero_grad()
        trace_potential.backward(retain_graph=True)
        with torch.no_grad():
            for param, grad in zip(net.parameters(), self.grads, strict=True):
                grad.data.copy_(param.grad if param.grad is not None else torch.zeros_like(param))

        # Pass 2 : if a previous state exists, update params.
        if last_state is not None:
            net.zero_grad()
            entropy.backward()
            with torch.no_grad():
                v_last = net(last_state)[1]
                delta = cfg.gamma * (0.0 if is_terminal else v_curr) + reward - v_last

                for param, trace, msg in zip(net.parameters(), self.traces, self.msg, strict=True):
                    param_grad = param.grad if param.grad is not None else torch.zeros_like(param)
                    # AC(λ) gradient : eligibility-trace-weighted TD error plus
                    # entropy bonus contribution.
                    g = trace * delta[0] + cfg.beta * param_grad
                    # Custom RMSprop with initialization debiasing.
                    msg.copy_(cfg.gamma_rms * msg + (1 - cfg.gamma_rms) * g * g)
                    denom = torch.sqrt(msg / (1 - cfg.gamma_rms ** (time_step + 1)) + cfg.eps_rms)
                    param.copy_(param + cfg.alpha * g / denom)

        # Always update eligibility traces.
        with torch.no_grad():
            for grad, trace in zip(self.grads, self.traces, strict=True):
                trace.copy_(cfg.lambda_ * cfg.gamma * trace + grad)

    def train(self) -> dict:
        """Run the full training budget. Returns the summary dict that's also
        persisted to ``cfg.output_dir / 'summary.json'``."""
        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        t = 0          # frame counter
        e = 0          # episode counter
        avg_return = 0.0
        returns: list[float] = []
        frame_stamps: list[int] = []
        validation_returns: list[float] = []
        validation_returns_std: list[float] = []
        validation_frames: list[int] = []
        validation_episodes: list[int] = []

        t_start = time.perf_counter()

        while t < cfg.num_frames:
            G = 0.0
            self.env.reset()
            s = _get_state(self.env.state(), self.device)
            is_terminated = False
            s_last = None
            r_last = None
            term_last = None
            action = None

            while (not is_terminated) and t < cfg.num_frames:
                s_prime, action, reward, is_terminated_t = _world_step(s, self.env, self.network)
                is_terminated = bool(is_terminated_t.item())

                sample = Transition(s, s_last, action, r_last, term_last)
                self._update(sample, t)

                G += reward.item()
                t += 1

                s_last, r_last, term_last = s, reward, torch.tensor(
                    [[is_terminated]], device=self.device
                )
                s = s_prime

            # End-of-episode terminal update (reference does this too).
            if action is not None:
                sample = Transition(s, s_last, action, r_last, term_last)
                self._update(sample, t)

            # Clear traces between episodes.
            for trace in self.traces:
                trace.zero_()

            e += 1
            returns.append(G)
            frame_stamps.append(t)
            avg_return = 0.99 * avg_return + 0.01 * G

            if e % cfg.validation_every_episodes == 0:
                val_mean, val_std = _validation(
                    self.env, self.network, cfg.validation_episodes, self.device
                )
                validation_returns.append(val_mean)
                validation_returns_std.append(val_std)
                validation_frames.append(t)
                validation_episodes.append(e)
                log.info(
                    "[ACB %s seed=%d] ep=%d frame=%d/%d G=%.2f avg=%.2f val=%.2f±%.2f t/frame=%.4fs",
                    cfg.game, cfg.seed, e, t, cfg.num_frames, G, avg_return,
                    val_mean, val_std, (time.perf_counter() - t_start) / max(t, 1),
                )
            elif cfg.log_every_episodes and e % cfg.log_every_episodes == 0:
                log.info(
                    "[ACB %s seed=%d] ep=%d frame=%d/%d G=%.2f avg=%.2f",
                    cfg.game, cfg.seed, e, t, cfg.num_frames, G, avg_return,
                )

        elapsed = time.perf_counter() - t_start

        # Persist artifacts.
        torch.save(self.network.state_dict(), cfg.output_dir / "policy_value.pt")
        np.save(cfg.output_dir / "returns.npy", np.asarray(returns, dtype=np.float32))
        np.save(cfg.output_dir / "frame_stamps.npy", np.asarray(frame_stamps, dtype=np.int64))
        np.save(
            cfg.output_dir / "validation_returns.npy",
            np.asarray(validation_returns, dtype=np.float32),
        )
        np.save(
            cfg.output_dir / "validation_returns_std.npy",
            np.asarray(validation_returns_std, dtype=np.float32),
        )
        np.save(
            cfg.output_dir / "validation_frames.npy",
            np.asarray(validation_frames, dtype=np.int64),
        )

        summary = {
            "setting": "setting-7-acb",
            "algorithm": "ACB",  # Young & Tian 2019 AC(λ)
            "game": cfg.game,
            "seed": cfg.seed,
            "num_frames": cfg.num_frames,
            "num_episodes": e,
            "elapsed_seconds": elapsed,
            "final_return": float(returns[-1]) if returns else 0.0,
            "avg_return_ema": float(avg_return),
            "last_validation_mean": float(validation_returns[-1]) if validation_returns else float("nan"),
            "last_validation_std": float(validation_returns_std[-1]) if validation_returns_std else float("nan"),
            "n_validation_points": len(validation_returns),
            "hyperparameters": {
                "alpha": cfg.alpha,
                "lambda": cfg.lambda_,
                "gamma": cfg.gamma,
                "beta": cfg.beta,
                "gamma_rms": cfg.gamma_rms,
                "eps_rms": cfg.eps_rms,
                "min_denom": cfg.min_denom,
            },
        }
        (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info(
            "[ACB %s seed=%d] DONE %d frames in %.1fs — final G=%.2f, val mean=%.2f",
            cfg.game, cfg.seed, t, elapsed,
            summary["final_return"], summary["last_validation_mean"],
        )
        return summary
